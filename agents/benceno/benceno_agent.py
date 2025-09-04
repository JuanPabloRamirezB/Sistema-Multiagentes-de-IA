# # Importar las librerías necesarias
# import os
# import json
# import requests
# from pymongo import MongoClient
# from pymongo.errors import ConnectionFailure
# from tqdm import tqdm # Para barras de progreso visuales

# from llama_index.core import (
#     VectorStoreIndex,
#     Settings,
#     StorageContext,
#     Document
# )
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.tools import FunctionTool

# # Importar las clases de Cohere y HuggingFace para LlamaIndex
# from llama_index.llms.cohere import Cohere
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# # Importar ChatMessage y MessageRole para un manejo robusto del historial de chat
# from llama_index.core.llms import ChatMessage, MessageRole

# from llama_index.vector_stores.chroma import ChromaVectorStore
# import chromadb

# import asyncio

# # --- Configuración global de LlamaIndex ---
# # Configurar el modelo de embeddings (HuggingFace)
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# # Configurar el LLM (Cohere)
# # Asegúrate de que tu clave API de Cohere esté configurada como variable de entorno
# # o aquí directamente si es para pruebas locales y no un entorno de producción.
# # os.environ["COHERE_API_KEY"] = "djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"
# # Settings.llm = Cohere(model="command-r", api_key=os.environ["COHERE_API_KEY"])
# # Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
# Settings.llm = Ollama(
#     model="llama3.2",
#     request_timeout=360.0,
#     base_url="http://localhost:11434"
# )
# # ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# # Settings.llm = Ollama(model="mistral-openorca", base_url=ollama_url, request_timeout=600)

# print(f"LlamaIndex configurado con LLM (Cohere: command-r) y Embeddings (HuggingFace: BAAI/bge-m3).")

# # --- Configuración de Conexión a MongoDB (para datos fuente y chat history) ---
# MONGO_URI = "mongodb://localhost:27017/" # URI para tu contenedor Docker de MongoDB
# # Base de datos donde los datos de la API de Benceno fueron almacenados (la base de datos fuente)
# DB_NAME_SOURCE = "benceno"
# # Nueva base de datos para el historial de chat (ChromaDB manejará los embeddings)
# DB_NAME_CHAT_HISTORY = "benceno_chat_history_db" # Renombrado para claridad
# # Nueva colección para el historial del chat
# COLLECTION_NAME_CHAT_HISTORY = "chat_history_benceno"


# print("1. Conectando a MongoDB para fuente de datos e historial de chat...")
# try:
#     mongo_client = MongoClient(MONGO_URI)
#     mongo_client.admin.command('ping') # Prueba la conexión
#     print("Conexión a MongoDB exitosa!")
# except ConnectionFailure as e:
#     print(f"Error al conectar a MongoDB: {e}")
#     print("Asegúrate de que tu contenedor de MongoDB esté corriendo. Ejecuta 'docker run -d --name my-mongo -p 27017:27017 -v mongo-data:/data/db mongo:latest' en tu terminal.")
#     exit() 

# # Referencias a las bases de datos de origen y de historial de chat
# db_source = mongo_client[DB_NAME_SOURCE]
# db_chat_history = mongo_client[DB_NAME_CHAT_HISTORY] # Nueva referencia
# mongo_chat_history_collection = db_chat_history[COLLECTION_NAME_CHAT_HISTORY] # La colección para el historial de chat

# # --- Configuración de ChromaDB ---
# # Conectarse a la instancia de ChromaDB en Docker
# # Se usa HttpClient para apuntar al servidor de ChromaDB
# CHROMA_HOST = "localhost"
# CHROMA_PORT = 8000
# COLLECTION_NAME_CHROMA = "emisiones_benceno_chroma_collection" # Nombre de la colección en ChromaDB

# print(f"\n2. Conectando a ChromaDB en http://{CHROMA_HOST}:{CHROMA_PORT}...")
# try:
#     # Cliente persistente si quieres guardar en disco, o cliente HTTP para Docker
#     chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
#     # Intentar obtener la colección; si no existe, la crea
#     # ChromaDB crea la colección si no existe al llamarla por primera vez.
#     # No hay una función directa para "ping" HTTP en chromadb como en pymongo
#     # Una forma de verificar sería intentar listar colecciones o crear una.
#     print(f"Conexión a ChromaDB establecida. Usando colección: {COLLECTION_NAME_CHROMA}")
# except Exception as e:
#     print(f"Error al conectar a ChromaDB: {e}")
#     print("Asegúrate de que tu contenedor de ChromaDB esté corriendo. Ejecuta el 'docker run' apropiado.")
#     exit()

# # Crear Vector Store usando ChromaDB
# vector_store = ChromaVectorStore(
#     chroma_collection=chroma_client.get_or_create_collection(COLLECTION_NAME_CHROMA)
# )

# # Crear StorageContext con ChromaDB
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# # --- Función para cargar datos desde la base de datos 'benceno' de MongoDB ---
# def load_data_from_mongo_benceno():
#     """
#     Carga documentos de emisiones de benceno desde la base de datos 'benceno' en MongoDB,
#     transformándolos en objetos Document de LlamaIndex.
#     """
#     print(f"\n3. Cargando y procesando datos desde la base de datos '{DB_NAME_SOURCE}'...")
#     llama_documents = []

#     try:
#         source_collection_names = db_source.list_collection_names()
#         if not source_collection_names:
#             print(f"La base de datos '{DB_NAME_SOURCE}' está vacía o no existe. Por favor, asegúrate de haber ejecutado el script 'descargar_benceno.py' primero.")
#             return [] # Retorna lista vacía si no hay datos
#     except Exception as e:
#         print(f"Error al listar colecciones en la base de datos '{DB_NAME_SOURCE}': {e}")
#         print("Asegúrate de que la base de datos 'benceno' haya sido creada y poblada.")
#         return [] # Retorna lista vacía en caso de error

#     print(f"Colecciones (estados) encontradas en '{DB_NAME_SOURCE}': {len(source_collection_names)}")

#     # Itera sobre cada colección (estado) en la base de datos 'benceno'
#     for collection_name in tqdm(source_collection_names, desc="Procesando estados desde MongoDB"):
#         collection = db_source[collection_name]
        
#         # Itera sobre cada documento (municipio) en la colección actual
#         for municipio_doc in collection.find({}):
#             # Extraer metadatos comunes para todos los años de este municipio
#             estado_cve_ent = municipio_doc.get("estado_cve_ent")
#             estado_nombre = municipio_doc.get("estado_nombre")
#             sustancia_cas = municipio_doc.get("sustancia_cas")
#             sustancia_nombre = municipio_doc.get("sustancia_nombre")
#             municipio_cve_mun = municipio_doc.get("municipio_cve_mun")
#             municipio_nombre = municipio_doc.get("municipio_nombre")

#             # Iterar sobre los datos de cada año dentro del municipio
#             for anio_data in municipio_doc.get("data", []):
#                 anio = anio_data.get("anio")
#                 cantidad_kg = anio_data.get("cantidad_kg")
#                 index_kg = anio_data.get("index_kg")

#                 # Crear un texto descriptivo para el documento de LlamaIndex
#                 text_content = (
#                     f"En el estado de {estado_nombre} (clave {estado_cve_ent}), "
#                     f"municipio {municipio_nombre} (clave {municipio_cve_mun}), "
#                     f"para el año {anio}, "
#                     f"la cantidad de emisiones de {sustancia_nombre} fue de {cantidad_kg:.2f} kg. "
#                     f"El índice de exposición para esta cantidad fue de {index_kg:.4f}."
#                 )

#                 # Crear metadatos detallados para el documento de LlamaIndex
#                 metadata = {
#                     "estado_cve_ent": estado_cve_ent,
#                     "estado_nombre": estado_nombre,
#                     "municipio_cve_mun": municipio_cve_mun,
#                     "municipio_nombre": municipio_nombre,
#                     "sustancia_cas": sustancia_cas,
#                     "sustancia_nombre": sustancia_nombre,
#                     "anio": anio,
#                     "cantidad_kg": cantidad_kg,
#                     "index_kg": index_kg,
#                     "umbral_nom_eyt_excede": anio_data.get("umbral_nom_eyt", {}).get("excede"),
#                     "umbral_nom_mpu_excede": anio_data.get("umbral_nom_mpu", {}).get("excede"),
#                 }
#                 llama_documents.append(Document(text=text_content, metadata=metadata))
    
#     print(f"Se cargó un total de {len(llama_documents)} documentos de LlamaIndex desde MongoDB.")
#     return llama_documents

# # --- 4. Verificar e Indexar Datos en ChromaDB ---
# print("\n4. Verificando el estado del índice vectorial en ChromaDB...")

# # Verificar si la colección de ChromaDB ya tiene datos.
# # Necesitamos una forma de saber si ya se indexaron documentos.
# # Esto se hace consultando el conteo de la colección.
# try:
#     # Obtenemos la colección directamente desde el cliente de ChromaDB
#     chroma_collection_actual = chroma_client.get_or_create_collection(COLLECTION_NAME_CHROMA)
#     collection_count = chroma_collection_actual.count()

#     if collection_count > 0:
#         print(f"⚡ Conectado a un índice vectorial existente en ChromaDB con {collection_count} documentos. Cargando índice...")
#         index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#     else:
#         print("📄 La colección de ChromaDB está vacía. Indexando documentos nuevos...")
#         documents = load_data_from_mongo_benceno()
#         if not documents:
#             print("No hay documentos para indexar. Asegúrate de que la base de datos 'benceno' esté poblada.")
#             exit()
#         index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#         print("✅ Documentos indexados en ChromaDB.")
# except Exception as e:
#     print(f"Error al verificar/cargar el índice en ChromaDB: {e}")
#     print("Asegúrate de que ChromaDB esté accesible y la colección exista o pueda ser creada.")
#     exit()

# # Crear un QueryEngine para la búsqueda de similitud
# query_engine = index.as_query_engine(similarity_top_k=3)

# # --- Definición de Herramientas para el Agente ---
# async def search_benceno_info(query: str) -> str:
#     """
#     Busca información sobre emisiones de benceno en los documentos indexados.
#     Utiliza esta herramienta para cualquier pregunta sobre cantidades de benceno,
#     municipios, estados o años de emisión.
#     """
#     print(f"\n💡 Agente utilizando 'search_benceno_info' para la consulta: '{query}'")
#     response = query_engine.query(query)
    
#     # Formatear la respuesta para el agente
#     if response and response.response:
#         sources_info = "\n".join([
#             f"- {node.metadata.get('municipio_nombre', 'N/A')}, {node.metadata.get('estado_nombre', 'N/A')} ({node.metadata.get('anio', 'N/A')}): {node.text.strip()}"
#             for node in response.source_nodes
#         ])
#         return (f"Respuesta basada en datos indexados:\n{response.response}\n\n"
#                 f"Fuentes relevantes:\n{sources_info}")
#     return "No se encontró información relevante en los documentos indexados."

# # Puedes mantener esta herramienta si el agente necesita realizar cálculos simples
# def multiply(a: float, b: float) -> float:
#     """Multiplies two numbers."""
#     return a * b

# # Crear herramientas con FunctionTool
# # search_tool = FunctionTool.from_defaults(fn=search_benceno_info, name="search_benceno_info")
# # multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply_numbers")


# # --- Creación del Agente ---
# # agent_worker = FunctionCallingAgentWorker.from_tools(
# #     tools=[multiply_tool, search_tool], # Se incluyen ambas herramientas para demostrar la capacidad del agente
# #     llm=Settings.llm, # El LLM globalmente configurado (Cohere command-r)
# #     system_prompt = """Eres un asistente experto en información técnica sobre emisiones de sustancias químicas, específicamente Benceno, por municipio y año en México.
# # Tu función principal es responder preguntas basándote en los datos disponibles que has indexado.
# # Cuando necesites encontrar información sobre cantidades de benceno, municipios, estados o años de emisión, DEBES usar la herramienta 'search_benceno_info'.
# # Si la pregunta involucra cálculos numéricos simples, puedes usar la herramienta 'multiply_numbers'.
# # Al responder, sé preciso, incluye detalles técnicos relevantes (como cantidades, años, estados, municipios, índices si son pertinentes) y formatea la información de manera clara.
# # Después de proporcionar la información detallada y las fuentes, ofrece un resumen claro y conciso de la información solicitada.
# # Si la pregunta no se relaciona con la información disponible o las herramientas, házmelo saber.
# # """,
# # )
# # agent = agent_worker.as_agent()

# # --- Funciones para manejar el historial de chat en MongoDB ---

# # Agent builder
# def get_benceno_agent():
#     search_tool = FunctionTool.from_defaults(fn=search_benceno_info, name="search_benceno_info")
#     multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply_numbers")
#     return FunctionCallingAgentWorker.from_tools(
#         tools=[search_tool, multiply_tool],
#         llm=Settings.llm,
#         system_prompt = """Eres un asistente experto en información técnica sobre emisiones de sustancias químicas, específicamente Benceno, por municipio y año en México.
# Tu función principal es responder preguntas basándote en los datos disponibles que has indexado.
# Cuando necesites encontrar información sobre cantidades de benceno, municipios, estados o años de emisión, DEBES usar la herramienta 'search_benceno_info'.
# Si la pregunta involucra cálculos numéricos simples, puedes usar la herramienta 'multiply_numbers'.
# Al responder, sé preciso, incluye detalles técnicos relevantes (como cantidades, años, estados, municipios, índices si son pertinentes) y formatea la información de manera clara.
# Después de proporcionar la información detallada y las fuentes, ofrece un resumen claro y conciso de la información solicitada.
# Si la pregunta no se relaciona con la información disponible o las herramientas, házmelo saber.
# """,
#     )

# import os
# import logging
# from typing import Optional, List, Dict

# import chromadb
# from pymongo import MongoClient

# from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
# from llama_index.core.node_parser import SimpleNodeParser
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.cohere import Cohere
# from llama_index.llms.ollama import Ollama
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.tools import FunctionTool

# # === Logging ===
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # === SafeOllama: parche para evitar NoneType en tool-calls (si lo necesitas) ===
# class SafeOllama(Ollama):
#     def get_tool_calls_from_response(self, *args, **kwargs):
#         try:
#             tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
#         except Exception:
#             return []
#         return tool_calls or []

# # === Embeddings global (ajusta modelo si lo deseas) ===
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")


# def connect_mongo(mongo_uri: str) -> MongoClient:
#     client = MongoClient(mongo_uri)
#     logger.info("✅ Conectado a MongoDB")
#     return client


# def connect_chroma(collection_name: str, host: str = "0.0.0.0", port: int = 8000):
#     chroma_client = chromadb.HttpClient(host=host, port=port)
#     chroma_collection = chroma_client.get_or_create_collection(collection_name)
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     logger.info(f"✅ Conectado a Chroma colección: {collection_name}")
#     return chroma_client, chroma_collection, vector_store, storage_context


# def load_data_from_mongo(client: MongoClient, db_name: str,
#                          collection_name: Optional[str] = None,
#                          field_map: Optional[Dict[str, str]] = None,
#                          limit: Optional[int] = 1000) -> List[Document]:
#     """
#     Devuelve una lista de Document (LlamaIndex) desde MongoDB.
#     Si collection_name es None, carga todas las colecciones de la base.
#     field_map: opcional para mapear campos -> etiquetas.
#     limit: límite total aproximado de documentos leídos (para no saturar).
#     """
#     db = client[db_name]
#     docs: List[Document] = []
#     read = 0

#     collections = [collection_name] if collection_name else db.list_collection_names()

#     for col in collections:
#         collection = db[col]
#         cursor = collection.find({})
#         for raw in cursor:
#             if read >= (limit or 1000):
#                 break
#             # Construir texto descriptivo según field_map o todo el documento
#             if field_map:
#                 parts = []
#                 for k, label in field_map.items():
#                     if k in raw:
#                         parts.append(f"{label}: {raw.get(k)}")
#                 text = " | ".join(parts)
#             else:
#                 # quitar _id para legibilidad
#                 raw_copy = {k: v for k, v in raw.items() if k != "_id"}
#                 text = json_safe_str(raw_copy)

#             metadata = {"source_collection": col}
#             docs.append(Document(text=text, metadata=metadata))
#             read += 1
#         if read >= (limit or 1000):
#             break

#     logger.info(f"📥 Cargados {len(docs)} documentos desde MongoDB (db={db_name})")
#     return docs


# def json_safe_str(obj) -> str:
#     # Convierte a string sin problemas con tipos complejos (ObjectId, datetimes...)
#     try:
#         import json
#         return json.dumps(obj, default=str, ensure_ascii=False)
#     except Exception:
#         return str(obj)


# def get_generic_mongo_agent(
#     mongo_uri: str,
#     db_name: str,
#     chroma_collection_name: str,
#     field_map: Optional[Dict[str, str]] = None,
#     llm_choice: str = "cohere",
#     llm_params: Optional[Dict[str, str]] = None,
#     mongo_collection_name: Optional[str] = None,
#     min_results_threshold: int = 1,
#     mongo_read_limit: int = 500,
# ):
#     """
#     Crea un agente que:
#      - intenta buscar primero en Chroma,
#      - si no hay suficientes resultados (< min_results_threshold) consulta Mongo,
#        vectoriza/guarda en Chroma y reconsulta.
#     Parámetros:
#      - mongo_collection_name: si se especifica, solo lee esa colección; si no, lee todas.
#      - min_results_threshold: número mínimo de source_nodes para considerar "suficiente".
#      - mongo_read_limit: número máximo de documentos a leer desde Mongo por actualización.
#     """

#     # conexiones
#     client = connect_mongo(mongo_uri)
#     chroma_client, chroma_collection, vector_store, storage_context = connect_chroma(chroma_collection_name)

#     # Configurar LLM dinámico (y asignarlo a Settings.llm para evitar errores de fallback)
#     llm = None
#     llm_params = llm_params or {}
#     if llm_choice.lower() == "ollama":
#         Settings.llm = SafeOllama(
#             model=llm_params.get("model", "llama3.2"),
#             base_url=llm_params.get("base_url_API_key", "http://localhost:11434"),
#             request_timeout=360.0
#         )
#         logger.info("✅ Usando Ollama como LLM")
#     elif llm_choice.lower() == "cohere":
#         api_key = llm_params.get("base_url_API_key")
#         Settings.llm = Cohere(model=llm_params.get("model", "command-r"), api_key=api_key)
#         logger.info("✅ Usando Cohere como LLM")
#     else:
#         raise ValueError("LLM inválido. Debe ser 'ollama' o 'cohere'")

#     # también lo colocamos en Settings para que LlamaIndex no intente resolver otro LLM por defecto
#     # Settings.llm = llm

#     # Construir (o cargar) índice inicial desde Chroma
#     try:
#         collection_count = chroma_collection.count()
#     except Exception:
#         # En algunos entornos/counts puede fallar; asumir 0
#         collection_count = 0

#     if collection_count > 0:
#         logger.info(f"🔍 Chroma ya contiene {collection_count} vectores. Usando índice existente.")
#         index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#     else:
#         logger.info("⚠️ Chroma vacío — inicializando índice vacío (se llenará si hace falta).")
#         index = VectorStoreIndex([], storage_context=storage_context)

#     node_parser = SimpleNodeParser.from_defaults()

#     # ---- hybrid_search como closure para acceder a client, db_name, etc. ----
#     def hybrid_search(query: str, top_k: int = 3) -> str:
#         """
#         1) consulta Chroma
#         2) si resultados < min_results_threshold -> lee de Mongo, indexa y reconsulta
#         Retorna texto (respuesta generada por query_engine.response).
#         """
#         nonlocal index  # porque podemos re-crear o insertar nodos en el índice

#         # 1) consultar en Chroma (índice actual)
#         query_engine = index.as_query_engine(similarity_top_k=top_k)
#         chroma_resp = query_engine.query(query)

#         # si hay resultados suficientes, devolverlos
#         num_sources = len(getattr(chroma_resp, "source_nodes", []) or [])
#         logger.info(f"🔎 Búsqueda en Chroma, resultados: {num_sources}")
#         if num_sources >= min_results_threshold and getattr(chroma_resp, "response", None):
#             return format_response_with_sources(chroma_resp)

#         # 2) no hay suficientes -> leer desde Mongo (colección específica o todas)
#         logger.info("📥 No hay suficientes resultados en Chroma -> consultando Mongo...")
#         docs_to_index = load_data_from_mongo(client, db_name, collection_name=mongo_collection_name,
#                                             field_map=field_map, limit=mongo_read_limit)
#         if not docs_to_index:
#             logger.warning("❌ No se encontraron documentos en Mongo para indexar.")
#             return "No encontré información ni en Chroma ni en Mongo."

#         # convertir Document list en nodos e insertarlos en el índice (y Chroma)
#         nodes = node_parser.get_nodes_from_documents(docs_to_index)
#         logger.info(f"📤 Indexando {len(nodes)} nodos en Chroma (mediante LlamaIndex)...")
#         index.insert_nodes(nodes)  # esto añadirá a la store (Chroma) a través del storage_context/vector_store

#         # reconstruir query engine y reconsultar
#         query_engine = index.as_query_engine(similarity_top_k=top_k)
#         final_resp = query_engine.query(query)
#         logger.info(f"🔁 Reconsulta tras indexar -> resultados: {len(getattr(final_resp, 'source_nodes', []) or [])}")
#         if getattr(final_resp, "response", None):
#             return format_response_with_sources(final_resp)

#         # fallback
#         return "No se pudo encontrar información útil después de indexar datos desde Mongo."

#     # Herramienta LlamaIndex para el agente
#     hybrid_tool = FunctionTool.from_defaults(
#         fn=hybrid_search,
#         name="hybrid_search",
#         description="Busca primero en Chroma; si no hay suficientes resultados consulta Mongo, indexa y reintenta."
#     )

#     # Crear y devolver agente (FunctionCallingAgentWorker)
#     system_prompt = f"""Eres un asistente que responde preguntas sobre los datos en la base de datos '{db_name}'.
# Cuando necesites buscar información, usa la herramienta 'hybrid_search'. Responde con precisión y cita las fuentes.
# Siempre responde en español, incluso si los datos vienen en otro idioma."""
#     agent_worker = FunctionCallingAgentWorker.from_tools(
#         tools=[hybrid_tool],
#         llm=Settings.llm,
#         system_prompt=system_prompt,
#         verbose=True
#     )
#     return agent_worker


# # -----------------------
# # Helpers
# # -----------------------
# def format_response_with_sources(resp) -> str:
#     """
#     Formatea una respuesta de query_engine añadiendo una lista compacta de fuentes.
#     """
#     text = getattr(resp, "response", "") or ""
#     source_nodes = getattr(resp, "source_nodes", []) or []
#     if not source_nodes:
#         return text
#     sources_lines = []
#     for n in source_nodes:
#         meta = getattr(n, "metadata", {}) or {}
#         # intenta mostrar colección/metadata si está disponible
#         col = meta.get("source_collection") or meta.get("collection") or meta
#         sources_lines.append(f"- {col}")
#     return f"{text}\n\nFuentes:\n" + "\n".join(sources_lines)
#######################################################################################################################3
# main.py
import os
import logging
from typing import Optional, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

import chromadb
from pymongo import MongoClient

from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.llms.ollama import Ollama
# from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent.workflow import FunctionAgent 
from llama_index.core.tools import FunctionTool

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Safe Ollama
class SafeOllama(Ollama):
    def get_tool_calls_from_response(self, *args, **kwargs):
        try:
            tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
        except Exception:
            return []
        return tool_calls or []

# Embeddings global
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")


def connect_mongo(mongo_uri: str) -> MongoClient:
    client = MongoClient(mongo_uri)
    logger.info("✅ Conectado a MongoDB")
    return client


def connect_chroma(collection_name: str, host: str = "0.0.0.0", port: int = 8000):
    chroma_client = chromadb.HttpClient(host=host, port=port)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info(f"✅ Conectado a Chroma colección: {collection_name}")
    return chroma_client, chroma_collection, vector_store, storage_context


def json_safe_str(obj) -> str:
    import json
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


def load_data_from_mongo(client: MongoClient, db_name: str,
                         collection_name: Optional[str] = None,
                         field_map: Optional[Dict[str, str]] = None,
                         limit: Optional[int] = 500) -> List[Document]:
    db = client[db_name]
    docs: List[Document] = []
    read = 0
    collections = [collection_name] if collection_name else db.list_collection_names()

    for col in collections:
        collection = db[col]
        cursor = collection.find({})
        for raw in cursor:
            if read >= (limit or 500):
                break
            if field_map:
                parts = []
                for k, label in field_map.items():
                    if k in raw:
                        parts.append(f"{label}: {raw.get(k)}")
                text = " | ".join(parts)
            else:
                raw_copy = {k: v for k, v in raw.items() if k != "_id"}
                text = json_safe_str(raw_copy)
            metadata = {"source_collection": col}
            docs.append(Document(text=text, metadata=metadata))
            read += 1
        if read >= (limit or 500):
            break
    logger.info(f"📥 Cargados {len(docs)} documentos desde MongoDB (db={db_name})")
    return docs


def get_generic_mongo_agent(
    mongo_uri: str,
    db_name: str,
    chroma_collection_name: str,
    field_map: Optional[Dict[str, str]] = None,
    llm_choice: str = "cohere",
    llm_params: Optional[Dict[str, str]] = None,
    mongo_collection_name: Optional[str] = None,
    min_results_threshold: int = 1,
    mongo_read_limit: int = 500,
):
    client = connect_mongo(mongo_uri)
    chroma_client, chroma_collection, vector_store, storage_context = connect_chroma(chroma_collection_name)
    node_parser = SimpleNodeParser.from_defaults()

    # Configuración LLM
    llm_params = llm_params or {}
    if llm_choice.lower() == "ollama":
        Settings.llm = SafeOllama(
            model=llm_params.get("model", "llama3.2"),
            base_url=llm_params.get("base_url_API_key", "http://localhost:11434"),
            request_timeout=360.0
        )
    elif llm_choice.lower() == "cohere":
        api_key = llm_params.get("base_url_API_key", os.getenv("COHERE_API_KEY"))
        Settings.llm = Cohere(model=llm_params.get("model", "command-r"), api_key=api_key)
    else:
        raise ValueError("LLM inválido")

    # Índice inicial
    try:
        collection_count = chroma_collection.count()
    except Exception:
        collection_count = 0

    if collection_count > 0:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        index = VectorStoreIndex([], storage_context=storage_context)

    # Closure para la búsqueda híbrida
    # def hybrid_search(query: str, top_k: int = 3) -> str:
    #     nonlocal index
    #     if len(index.nodes) == 0:
    #         docs_to_index = load_data_from_mongo(client, db_name, collection_name=mongo_collection_name,
    #                                             field_map=field_map, limit=mongo_read_limit)
    #         if docs_to_index:
    #             nodes = node_parser.get_nodes_from_documents(docs_to_index)
    #             index.insert_nodes(nodes)
    #         else:
    #             return "No hay datos disponibles en Mongo ni en Chroma."

    #     query_engine = index.as_query_engine(similarity_top_k=top_k)
    #     resp = query_engine.query(query)
    #     if getattr(resp, "response", None):
    #         return format_response_with_sources(resp)
    #     else:
    #         return "No se encontró información relevante."
    def hybrid_search(query: str, top_k: int = 3) -> str:
        nonlocal index

        # Revisar si Chroma tiene vectores
        try:
            vector_count = chroma_collection.count()
        except Exception:
            vector_count = 0

        if vector_count == 0:
            # No hay datos en Chroma -> leer Mongo y crear nodos
            docs_to_index = load_data_from_mongo(client, db_name, collection_name=mongo_collection_name,
                                                field_map=field_map, limit=mongo_read_limit)
            if not docs_to_index:
                return "No hay datos disponibles ni en Chroma ni en Mongo."
            nodes = node_parser.get_nodes_from_documents(docs_to_index)
            index.insert_nodes(nodes)  # esto indexa en Chroma
            logger.info(f"📤 Indexados {len(nodes)} nodos en Chroma desde Mongo.")

        # Consultar índice
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        resp = query_engine.query(query)

        # Revisar si hubo resultados
        num_sources = len(getattr(resp, "source_nodes", []) or [])
        if num_sources < min_results_threshold:
            # Leer más datos de Mongo si es necesario
            docs_to_index = load_data_from_mongo(client, db_name, collection_name=mongo_collection_name,
                                                field_map=field_map, limit=mongo_read_limit)
            if docs_to_index:
                nodes = node_parser.get_nodes_from_documents(docs_to_index)
                index.insert_nodes(nodes)
                query_engine = index.as_query_engine(similarity_top_k=top_k)
                resp = query_engine.query(query)

        if getattr(resp, "response", None):
            return format_response_with_sources(resp)
        return "No se encontró información relevante."


    hybrid_tool = FunctionTool.from_defaults(
        fn=hybrid_search,
        name="hybrid_search",
        description="Busca primero en Chroma; si no hay suficientes resultados consulta Mongo y reintenta."
    )

    system_prompt = f"""Eres un asistente que responde preguntas sobre los datos en la base de datos '{db_name}'.
Cuando necesites buscar información, usa la herramienta 'hybrid_search'. Responde con precisión y cita las fuentes.
Siempre responde en español."""

    agent_worker = FunctionAgent(
        tools=[hybrid_tool],
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True,
        max_iterations=3
    )

    return agent_worker


def format_response_with_sources(resp) -> str:
    text = getattr(resp, "response", "") or ""
    source_nodes = getattr(resp, "source_nodes", []) or []
    if not source_nodes:
        return text
    sources_lines = []
    for n in source_nodes:
        meta = getattr(n, "metadata", {}) or {}
        col = meta.get("source_collection") or meta.get("collection") or meta
        sources_lines.append(f"- {col}")
    return f"{text}\n\nFuentes:\n" + "\n".join(sources_lines)


# # ---------------- FastAPI ----------------
# app = FastAPI()

# # Variables de entorno
# mongo_uri = os.getenv("MONGO_URI")
# db_name = os.getenv("DB_NAME")
# chroma_collection = os.getenv("CHROMA_COLLECTION")
# llm_choice = os.getenv("LLM_CHOICE", "cohere")
# model = os.getenv("MODEL")
# base_url_API_key = os.getenv("BASE_URL_API_KEY")

# agent = get_generic_mongo_agent(
#     mongo_uri=mongo_uri,
#     db_name=db_name,
#     chroma_collection_name=chroma_collection,
#     field_map=None,
#     llm_choice=llm_choice,
#     llm_params={"model": model, "base_url_API_key": base_url_API_key}
# ).as_agent()


# class QueryRequest(BaseModel):
#     query: str


# @app.post("/query")
# async def query_agent(req: QueryRequest):
#     response = await agent.achat(req.query)
#     return {"response": response.response}
