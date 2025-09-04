# import os
# import json
# import asyncio
# import logging
# import sys
# import traceback
# from typing import Optional, List, Dict

# from pymongo import MongoClient
# from pymongo.errors import ConnectionFailure

# from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.tools import FunctionTool
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.cohere import Cohere
# from llama_index.llms.ollama import Ollama
# from llama_index.vector_stores.chroma import ChromaVectorStore
# import chromadb
# from urllib.parse import urlparse

# # === Logging ===
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # === SafeOllama: parche para evitar NoneType ===
# class SafeOllama(Ollama):
#     def get_tool_calls_from_response(self, *args, **kwargs):
#         try:
#             tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
#         except Exception:
#             return []
#         return tool_calls or []

# # === Configuración global ===
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# # === Conexión a Mongo ===
# def connect_mongo(uri="mongodb://localhost:27017/") -> MongoClient:
#     try:
#         client = MongoClient(uri)
#         client.admin.command("ping")
#         logger.info("✅ Conexión a MongoDB exitosa")
#         return client
#     except ConnectionFailure as e:
#         raise RuntimeError(f"Error al conectar a MongoDB: {e}")

# # === Cargar datos de cualquier colección ===
# def load_data_from_mongo(
#     client: MongoClient,
#     db_name: str,
#     collections: Optional[List[str]] = None,
#     field_map: Optional[Dict[str, str]] = None
# ) -> List[Document]:
#     db = client[db_name]
#     if collections is None:
#         collections = db.list_collection_names()
    
#     llama_docs = []
#     for col_name in collections:
#         col = db[col_name]
#         for doc in col.find({}):
#             if field_map:
#                 text_parts = [f"{label}: {doc.get(field)}" for field, label in field_map.items()]
#                 text_content = ". ".join(text_parts)
#             else:
#                 text_content = json.dumps(doc, ensure_ascii=False, default=str)
#             llama_docs.append(Document(text=text_content, metadata={"collection": col_name}))
    
#     logger.info(f"📄 Se cargaron {len(llama_docs)} documentos desde MongoDB ({db_name})")
#     return llama_docs

# # === Conexión a Chroma ===
# def connect_chroma(collection_name: str, host, port):
#     chroma_client = chromadb.HttpClient(host=host, port=port)
#     vector_store = ChromaVectorStore(
#         chroma_collection=chroma_client.get_or_create_collection(collection_name)
#     )
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     logger.info("✅ Conexión a ChromaDB exitosa")
#     return vector_store, storage_context

# # === Herramienta de búsqueda ===
# def make_search_tool(query_engine):
#     def search_info(query: str) -> str:
#         response = query_engine.query(query)
#         if response and response.response:
#             sources = "\n".join([f"- {node.metadata}" for node in response.source_nodes])
#             return f"{response.response}\n\nFuentes:\n{sources}"
#         return "No se encontró información relevante."
#     return FunctionTool.from_defaults(fn=search_info, name="search_info")

# # === Construir agente genérico ===
# def get_generic_mongo_agent(
#     mongo_uri: str,
#     db_name: str,
#     chroma_uri:str,
#     chroma_collection_name: str,
#     field_map: Optional[Dict[str, str]] = None,
#     llm_choice: str = "cohere",
#     llm_params: Optional[Dict[str, str]] = None,
# ):
#     client = connect_mongo(mongo_uri)
#     chroma_uri_parsed = urlparse(chroma_uri)
#     host = chroma_uri_parsed.hostname
#     port = chroma_uri_parsed.port
#     vector_store, storage_context = connect_chroma(chroma_collection_name, host, port)

#     # Selección de LLM
#     if llm_choice.lower() == "ollama":
#         Settings.llm = SafeOllama(
#             model=llm_params.get("model"),
#             base_url=llm_params.get("base_url_API_key"),
#             request_timeout=360.0
#         )
#         logger.info("✅ Usando Ollama como LLM")
#     elif llm_choice.lower() == "cohere":
#         Settings.llm = Cohere(
#             model=llm_params.get("model"),
#             api_key=llm_params.get("base_url_API_key")
#         )
#         logger.info("✅ Usando Cohere como LLM")
#     else:
#         raise ValueError("LLM inválido. Debe ser 'ollama' o 'cohere'")

#     chroma_client = chromadb.HttpClient(host, port)
#     chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)

#     if chroma_collection.count() > 0:
#         index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#     else:
#         docs = load_data_from_mongo(client, db_name, field_map=field_map)
#         index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    
#     query_engine = index.as_query_engine(similarity_top_k=7)
#     search_tool = make_search_tool(query_engine)

#     return FunctionCallingAgentWorker.from_tools(
#         tools=[search_tool],
#         llm=Settings.llm,
#         verbose=True,
#         system_prompt=f"""Eres un asistente que responde preguntas sobre los datos en la base de datos '{db_name}'.
# Cuando necesites buscar información, usa la herramienta 'search_info'. Responde con precisión y cita las fuentes.
# Siempre responde en español, incluso si los datos vienen en otro idioma."""
#     )

import os
import json
import logging
import asyncio
from typing import Optional, List, Dict
from urllib.parse import urlparse

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from llama_index.core import (
    VectorStoreIndex, Settings, StorageContext, Document
)
# from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent.workflow import FunctionAgent 
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

# === Cargar datos de Mongo como Document ===
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
def connect_chroma(collection_name: str, host, port):
    chroma_client = chromadb.HttpClient(host=host, port=port)
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_client.get_or_create_collection(collection_name)
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("✅ Conexión a ChromaDB exitosa")
    return vector_store, storage_context

# === Herramienta de búsqueda con fallback ===
def make_fallback_tool(query_engine, client, db_name, storage_context, vector_store, field_map=None):
    def hybrid_search(query: str) -> str:
        try:
            # Paso 1: Intentar en Chroma
            response = query_engine.query(query)

            has_results = (
                response 
                and response.response 
                and hasattr(response, "source_nodes") 
                and len(response.source_nodes) > 0
            )

            if has_results:
                sources = "\n".join([f"- {node.metadata}" for node in response.source_nodes])
                return f"(🔎 ChromaDB)\n{response.response}\n\nFuentes:\n{sources}"

            logger.warning("⚠️ No se encontraron resultados en Chroma. Activando fallback a Mongo...")
        except Exception as e:
            logger.warning(f"⚠️ Error al consultar Chroma ({e}). Activando fallback a Mongo...")

        # Paso 2: Si falla, consultar en Mongo
        mongo_docs = load_data_from_mongo(client, db_name, field_map=field_map)
        if not mongo_docs:
            return "No se encontró información relevante en ninguna fuente."
        
        # Paso 3: Vectorizar y guardar en Chroma
        index_new = VectorStoreIndex.from_documents(mongo_docs, storage_context=storage_context)
        logger.info("⚡ Se agregaron documentos de Mongo a ChromaDB")
        
        # Paso 4: Volver a consultar Chroma
        try:
            new_engine = index_new.as_query_engine(similarity_top_k=7)
            response2 = new_engine.query(query)

            has_results2 = (
                response2 
                and response2.response 
                and hasattr(response2, "source_nodes") 
                and len(response2.source_nodes) > 0
            )

            if has_results2:
                sources = "\n".join([f"- {node.metadata}" for node in response2.source_nodes])
                return f"(📥 Mongo ➝ Chroma)\n{response2.response}\n\nFuentes:\n{sources}"
        except Exception as e:
            logger.warning(f"⚠️ Error al reintentar con Chroma ({e})")

        return "No se encontró información relevante."
    return FunctionTool.from_defaults(fn=hybrid_search, name="hybrid_search")


# === Construir agente genérico con Fallback RAG ===
def get_generic_mongo_agent(
    mongo_uri: str,
    db_name: str,
    chroma_uri:str,
    chroma_collection_name: str,
    field_map: Optional[Dict[str, str]] = None,
    llm_choice: str = "cohere",
    llm_params: Optional[Dict[str, str]] = None,
):
    client = connect_mongo(mongo_uri)
    chroma_uri_parsed = urlparse(chroma_uri)
    host = chroma_uri_parsed.hostname
    port = chroma_uri_parsed.port
    vector_store, storage_context = connect_chroma(chroma_collection_name, host, port)

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

    # Crear índice vacío si no hay nada en Chroma
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(similarity_top_k=7)

    search_tool = make_fallback_tool(query_engine, client, db_name, storage_context, vector_store, field_map)

    return FunctionAgent(
        tools=[search_tool],
        llm=Settings.llm,
        verbose=True,
        system_prompt=f"""Eres un asistente que responde preguntas sobre los datos en la base de datos '{db_name}'.
Cuando necesites buscar información, usa la herramienta 'search_tool'. 
Primero intenta en Chroma, y si no hay suficientes datos, busca en Mongo, indexa los resultados en Chroma y vuelve a buscar.
Siempre responde en español."""
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

    # agent = get_generic_mongo_agent(
    #     mongo_uri="mongodb://localhost:27017/",
    #     db_name="covid_db",
    #     chroma_uri="http://localhost:8000",
    #     chroma_collection_name="casos_covid",
    #     llm_choice="cohere",
    #     llm_params={
    #         # "model":"llama3.2",
    #         # "base_url_API_key":"http://localhost:11434"
    #         "model":"command-r",
    #         "base_url_API_key":"tEoaKGsnrDb7iS4HKF1v1dHAhPfYBXRSpzgEg8K2"
    #     }
    # ).as_agent()

    agent = get_generic_mongo_agent(
        mongo_uri="mongodb://localhost:27017/",
        db_name="benceno",
        chroma_uri="http://localhost:8000",
        chroma_collection_name="emisiones_benceno_chroma_collection",
        llm_choice="ollama",
        llm_params={
            "model":"gpt-oss:20b",
            "base_url_API_key":"http://148.247.204.68:11434"
            # "model":"command-r",
            # "base_url_API_key":"tEoaKGsnrDb7iS4HKF1v1dHAhPfYBXRSpzgEg8K2"
        }
    )

    print("🤖 Agente híbrido listo. Escribe 'salir' para terminar.")
    while True:
        user_input = input("\nUsuario > ")
        if user_input.lower() in {"salir", "exit"}:
            print("👋 Hasta luego.")
            break
        try:
            response = await agent.run(user_input)
            if not response:
                print("\n📡 Respuesta: El agente no devolvió nada.")
            elif getattr(response, "response", None):
                print(f"\n🤖 Agente > {response.response}")
            else:
                print("\n📡 Respuesta: No se pudo generar una respuesta clara.")
        except Exception as e:
            print(f"❌ Error: {e}")
            # traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    asyncio.run(main())
