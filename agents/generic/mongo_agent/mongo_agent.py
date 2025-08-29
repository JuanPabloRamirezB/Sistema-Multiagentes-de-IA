import os
import json
import asyncio
import logging
import sys
import traceback
from typing import Optional, List, Dict

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# === Logging ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === SafeOllama: parche para evitar NoneType ===
class SafeOllama(Ollama):
    def get_tool_calls_from_response(self, *args, **kwargs):
        try:
            tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
        except Exception:
            return []
        return tool_calls or []

# === Configuración global ===
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# === Conexión a Mongo ===
def connect_mongo(uri="mongodb://localhost:27017/") -> MongoClient:
    try:
        client = MongoClient(uri)
        client.admin.command("ping")
        logger.info("✅ Conexión a MongoDB exitosa")
        return client
    except ConnectionFailure as e:
        raise RuntimeError(f"Error al conectar a MongoDB: {e}")

# === Cargar datos de cualquier colección ===
def load_data_from_mongo(
    client: MongoClient,
    db_name: str,
    collections: Optional[List[str]] = None,
    field_map: Optional[Dict[str, str]] = None
) -> List[Document]:
    db = client[db_name]
    if collections is None:
        collections = db.list_collection_names()
    
    llama_docs = []
    for col_name in collections:
        col = db[col_name]
        for doc in col.find({}):
            if field_map:
                text_parts = [f"{label}: {doc.get(field)}" for field, label in field_map.items()]
                text_content = ". ".join(text_parts)
            else:
                text_content = json.dumps(doc, ensure_ascii=False, default=str)
            llama_docs.append(Document(text=text_content, metadata={"collection": col_name}))
    
    logger.info(f"📄 Se cargaron {len(llama_docs)} documentos desde MongoDB ({db_name})")
    return llama_docs

# === Conexión a Chroma ===
def connect_chroma(collection_name: str, host="chromadb", port=8000):
    chroma_client = chromadb.HttpClient(host=host, port=port)
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_client.get_or_create_collection(collection_name)
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("✅ Conexión a ChromaDB exitosa")
    return vector_store, storage_context

# === Herramienta de búsqueda ===
def make_search_tool(query_engine):
    def search_info(query: str) -> str:
        response = query_engine.query(query)
        if response and response.response:
            sources = "\n".join([f"- {node.metadata}" for node in response.source_nodes])
            return f"{response.response}\n\nFuentes:\n{sources}"
        return "No se encontró información relevante."
    return FunctionTool.from_defaults(fn=search_info, name="search_info")

# === Construir agente genérico ===
def get_generic_mongo_agent(
    mongo_uri: str,
    db_name: str,
    chroma_collection_name: str,
    field_map: Optional[Dict[str, str]] = None,
    llm_choice: str = "cohere",
    llm_params: Optional[Dict[str, str]] = None,
    
):
    client = connect_mongo(mongo_uri)
    vector_store, storage_context = connect_chroma(chroma_collection_name)

    # Selección de LLM
    if llm_choice.lower() == "ollama":
        Settings.llm = SafeOllama(
            model=llm_params.get("model"),
            base_url=llm_params.get("base_url_API_key"),
            request_timeout=360.0
        )
        logger.info("✅ Usando Ollama como LLM")
    elif llm_choice.lower() == "cohere":
        Settings.llm = Cohere(
            model=llm_params.get("model"),
            api_key=llm_params.get("base_url_API_key")
        )
        logger.info("✅ Usando Cohere como LLM")
    else:
        raise ValueError("LLM inválido. Debe ser 'ollama' o 'cohere'")

    chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)

    if chroma_collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        docs = load_data_from_mongo(client, db_name, field_map=field_map)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    search_tool = make_search_tool(query_engine)

    return FunctionCallingAgentWorker.from_tools(
        tools=[search_tool],
        llm=Settings.llm,
        system_prompt=f"""Eres un asistente que responde preguntas sobre los datos en la base de datos '{db_name}'.
Cuando necesites buscar información, usa la herramienta 'search_info'. Responde con precisión y cita las fuentes.
Siempre responde en español, incluso si los datos vienen en otro idioma."""
    )

# === Main asincrónico ===
async def main():

    # Elegir LLM aquí
    # llm_choice = input("LLM a usar (ollama/cohere): ").strip().lower()
    # if llm_choice == "ollama":
    #     llm_params = {
    #         "model": input("Modelo Ollama: ").strip(),
    #         "base_url": input("Base URL Ollama: ").strip()
    #     }
    # else:
    #     llm_params = {
    #         "model": input("Modelo Cohere: ").strip(),
    #         "api_key": input("API Key Cohere: ").strip()
    #     }

    agent = get_generic_mongo_agent(
        mongo_uri="mongodb://localhost:27017/",
        db_name="covid_db",
        chroma_collection_name="casos_covid",
        llm_choice="ollama",
        llm_params={
            "model":"llama3.2",
            "base_url_API_key":"http://localhost:11434"
        }
    ).as_agent()

    print("🤖 Agente híbrido listo. Escribe 'salir' para terminar.")
    while True:
        user_input = input("\nUsuario > ")
        if user_input.lower() in {"salir", "exit"}:
            print("👋 Hasta luego.")
            break
        try:
            response = await agent.achat(user_input)
            if not response:
                print("\n📡 Respuesta: El agente no devolvió nada.")
            elif getattr(response, "response", None):
                print(f"\n🤖 Agente > {response.response}")
            else:
                print("\n📡 Respuesta: No se pudo generar una respuesta clara.")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    asyncio.run(main())
