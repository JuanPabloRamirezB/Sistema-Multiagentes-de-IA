# import os
# import json
# import asyncio
# from typing import Optional, List, Dict

# from pymongo import MongoClient
# from pymongo.errors import ConnectionFailure

# from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.tools import FunctionTool
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.cohere import Cohere
# from llama_index.vector_stores.chroma import ChromaVectorStore
# import chromadb

# # === Configuración global ===
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
# os.environ["COHERE_API_KEY"] = "djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"
# Settings.llm = Cohere(model="command-r", api_key=os.environ["COHERE_API_KEY"])

# # === Conexión a Mongo ===
# def connect_mongo(uri="mongodb://localhost:27017/") -> MongoClient:
#     try:
#         client = MongoClient(uri)
#         client.admin.command('ping')
#         print("✅ Conexión a MongoDB exitosa")
#         return client
#     except ConnectionFailure as e:
#         raise RuntimeError(f"Error al conectar a MongoDB: {e}")

# # === Cargar datos de cualquier colección ===
# # def load_data_from_mongo(
# #     client: MongoClient,
# #     db_name: str,
# #     collections: Optional[List[str]] = None,
# #     field_map: Optional[Dict[str, str]] = None
# # ) -> List[Document]:
# #     """
# #     Carga documentos de cualquier base de datos de MongoDB y los transforma a Document de LlamaIndex.
# #     - field_map: opcional, mapea nombres de campos a etiquetas más descriptivas.
# #     """
# #     db = client[db_name]
# #     if collections is None:
# #         collections = db.list_collection_names()
    
# #     llama_docs = []
# #     for col_name in collections:
# #         col = db[col_name]
# #         for doc in col.find({}):
# #             # Transformar el documento a texto
# #             if field_map:
# #                 text_parts = [f"{label}: {doc.get(field)}" for field, label in field_map.items()]
# #                 text_content = ". ".join(text_parts)
# #             else:
# #                 text_content = json.dumps( doc, ensure_ascii=False)
            
# #             llama_docs.append(Document(text=text_content, metadata={"collection": col_name}))
    
# #     print(f"📄 Se cargaron {len(llama_docs)} documentos desde MongoDB ({db_name})")
# #     return llama_docs

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
#                 # Convertir automáticamente cualquier tipo raro (ObjectId, datetime, etc.)
#                 text_content = json.dumps(doc, ensure_ascii=False, default=str)
            
#             llama_docs.append(Document(text=text_content, metadata={"collection": col_name}))
    
#     print(f"📄 Se cargaron {len(llama_docs)} documentos desde MongoDB ({db_name})")
#     return llama_docs


# # === Conexión a Chroma ===
# def connect_chroma(collection_name: str, host="localhost", port=8000):
#     chroma_client = chromadb.HttpClient(host=host, port=port)
#     vector_store = ChromaVectorStore(
#         chroma_collection=chroma_client.get_or_create_collection(collection_name)
#     )
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     return vector_store, storage_context

# # === Herramienta de búsqueda ===
# def make_search_tool(query_engine):
#     def search_info(query: str) -> str:
#         response = query_engine.query(query)
#         if response and response.response:
#             sources = "\n".join(
#                 [f"- {node.metadata}" for node in response.source_nodes]
#             )
#             return f"{response.response}\n\nFuentes:\n{sources}"
#         return "No se encontró información relevante."
#     return FunctionTool.from_defaults(fn=search_info, name="search_info")

# # === Construir agente genérico ===
# def get_generic_mongo_agent(
#     mongo_uri: str,
#     db_name: str,
#     chroma_collection_name: str,
#     field_map: Optional[Dict[str, str]] = None
# ):
#     client = connect_mongo(mongo_uri)
#     vector_store, storage_context = connect_chroma(chroma_collection_name)

#     # Verificar si ya hay datos en Chroma
#     chroma_client = chromadb.HttpClient(host="localhost", port=8000)
#     chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)
#     if chroma_collection.count() > 0:
#         index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#     else:
#         docs = load_data_from_mongo(client, db_name, field_map=field_map)
#         index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
#         print(index)
    
#     query_engine = index.as_query_engine(similarity_top_k=3)

#     search_tool = make_search_tool(query_engine)

#     return FunctionCallingAgentWorker.from_tools(
#         tools=[search_tool],
#         llm=Settings.llm,
#         system_prompt=f"""Eres un asistente que responde preguntas sobre los datos en la base de datos '{db_name}'.
# Cuando necesites buscar información, usa la herramienta 'search_info'. Responde con precisión y cita las fuentes."""
#     )


# # === Main asincrónico ===
# async def main():
#     # Crear el agente genérico
#     agent = get_generic_mongo_agent(
#         mongo_uri="mongodb://localhost:27017/",
#         db_name="covid_db",
#         chroma_collection_name="casos_covid"
#     ).as_agent()
#     print("🤖 Agente híbrido listo. Escribe 'salir' para terminar.")

#     while True:
#         user_input = input("\nUsuario > ")
#         if user_input.lower() in {"salir", "exit"}:
#             print("👋 Hasta luego.")
#             break
#         try:
#             response = agent.chat(user_input)
#             print(f"\n🤖 Agente > {response.response}")
#         except Exception as e:
#             print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())
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

# === Logging global ===
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# === Configuración global ===
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
os.environ["COHERE_API_KEY"] = "djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"

# 💡 Clase SafeOllama (como en el otro agente)
class SafeOllama(Ollama):
    def get_tool_calls_from_response(self, *args, **kwargs):
        try:
            tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
        except TypeError:
            return []
        return tool_calls or []

# === Conexión a Mongo ===
def connect_mongo(uri="mongodb://localhost:27017/") -> MongoClient:
    try:
        client = MongoClient(uri)
        client.admin.command('ping')
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
def connect_chroma(collection_name: str, host="localhost", port=8000):
    chroma_client = chromadb.HttpClient(host=host, port=port)
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_client.get_or_create_collection(collection_name)
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_context

# === Herramienta de búsqueda ===
def make_search_tool(query_engine):
    def search_info(query: str) -> str:
        response = query_engine.query(query)
        if response and response.response:
            sources = "\n".join(
                [f"- {node.metadata}" for node in response.source_nodes]
            )
            return f"{response.response}\n\nFuentes:\n{sources}"
        return "No se encontró información relevante."
    return FunctionTool.from_defaults(fn=search_info, name="search_info")

# === Construir agente genérico ===
def get_generic_mongo_agent(
    mongo_uri: str,
    db_name: str,
    chroma_collection_name: str,
    field_map: Optional[Dict[str, str]] = None
):
    client = connect_mongo(mongo_uri)
    vector_store, storage_context = connect_chroma(chroma_collection_name)

    # Elegir LLM (SafeOllama preferido, fallback a Cohere)
    try:
        llm = SafeOllama(
            model="llama3.2",
            request_timeout=360.0,
            base_url="http://localhost:11434"
        )
        logger.info("Usando SafeOllama como LLM")
    except Exception as e:
        logger.warning(f"Error con Ollama, usando Cohere: {e}")
        llm = Cohere(model="command-r", api_key=os.environ["COHERE_API_KEY"])

    # Verificar si ya hay datos en Chroma
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)
    if chroma_collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        docs = load_data_from_mongo(client, db_name, field_map=field_map)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    search_tool = make_search_tool(query_engine)

    system_prompt = f"""Eres un asistente especializado en responder preguntas sobre la base de datos '{db_name}'.

Cuando el usuario pregunte:
1. Usa la herramienta 'search_info' para buscar datos relevantes.
2. Interpreta los resultados y respóndelos en español de forma clara.
3. Siempre incluye una referencia a las fuentes encontradas.

Si ocurre un error, explica qué pasó y sugiere posibles soluciones.
"""

    return FunctionCallingAgentWorker.from_tools(
        tools=[search_tool],
        llm=llm,
        system_prompt=system_prompt,
        verbose=True
    )

# === Main asincrónico ===
async def main():
    agent = get_generic_mongo_agent(
        mongo_uri="mongodb://localhost:27017/",
        db_name="covid_db",
        chroma_collection_name="casos_covid"
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
                print(f"\n📡 Respuesta:\n{response.response}")
            else:
                print("\n📡 Respuesta: No se pudo generar una respuesta clara.")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    asyncio.run(main())
