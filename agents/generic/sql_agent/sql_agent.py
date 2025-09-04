import os
import json
import asyncio
import logging
import sys
import traceback
from typing import Optional, List, Dict

import sqlalchemy
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.cohere import Cohere
import chromadb

# === Logging global ===
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 💡 Clase SafeOllama (manejo seguro de tool_calls)
class SafeOllama(Ollama):
    def get_tool_calls_from_response(self, *args, **kwargs):
        try:
            tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
        except TypeError:
            return []
        return tool_calls or []

# === Configuración global ===
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
# os.environ["COHERE_API_KEY"] = "djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"

# === Conexión a SQL ===
def connect_sql(sql_uri: str):
    try:
        engine = create_engine(sql_uri)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Conexión SQL exitosa")
        return engine
    except SQLAlchemyError as e:
        raise RuntimeError(f"Error al conectar a la base de datos SQL: {e}")

# === Cargar datos de todas las tablas ===
def load_data_from_sql(
    engine,
    tables: Optional[List[str]] = None,
    field_map: Optional[Dict[str, str]] = None
) -> List[Document]:
    insp = inspect(engine)

    if tables is None:
        tables = insp.get_table_names()

    llama_docs = []
    with engine.connect() as conn:
        for table in tables:
            columns = insp.get_columns(table)
            col_names = [col['name'] for col in columns]

            rows = conn.execute(text(f"SELECT * FROM {table}")).fetchall()

            for row in rows:
                row_dict = dict(zip(col_names, row))
                if field_map:
                    text_parts = [
                        f"{label}: {row_dict.get(field)}"
                        for field, label in field_map.items()
                    ]
                    text_content = ". ".join(text_parts)
                else:
                    text_content = json.dumps(row_dict, ensure_ascii=False, default=str)

                llama_docs.append(Document(text=text_content, metadata={"table": table}))

    logger.info(f"📄 Se cargaron {len(llama_docs)} registros desde {len(tables)} tablas")
    return llama_docs

# === Conexión a Chroma ===
def connect_chroma(collection_name: str, host="chromadb", port=8000):
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

        if response is None or response.response is None:
            return "No se encontró información relevante."

        sources = ""
        if response.source_nodes:
            sources = "\n".join([f"- {node.metadata}" for node in response.source_nodes])

        return f"{response.response}\n\nFuentes:\n{sources}" if sources else response.response

    return FunctionTool.from_defaults(fn=search_info, name="search_info")

# === Inserción en batch para evitar Payload too large ===
def batch_insert(index: VectorStoreIndex, docs: List[Document], batch_size: int = 200):
    node_parser = SimpleNodeParser.from_defaults()
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        nodes = node_parser.get_nodes_from_documents(batch)
        index.insert_nodes(nodes)
        logger.info(f"➡️ Insertados {i + len(batch)} / {len(docs)} documentos en Chroma")

# === Construir agente genérico para SQL ===
def get_generic_sql_agent(
    sql_uri: str,
    chroma_collection_name: str,
    field_map: Optional[Dict[str, str]] = None,
    llm_choice: str = "cohere",
    llm_params: Optional[Dict[str, str]] = None
):
    engine = connect_sql(sql_uri)
    insp = inspect(engine)

    vector_store, storage_context = connect_chroma(chroma_collection_name)

    # Configuración dinámica del LLM
    if llm_choice.lower() == "ollama":
        model = llm_params.get("model")
        base_url = llm_params.get("base_url")
        Settings.llm = SafeOllama(model=model, base_url=base_url, request_timeout=360.0)
    elif llm_choice.lower() == "cohere":
        model = llm_params.get("model")
        api_key = llm_params.get("api_key")
        os.environ["COHERE_API_KEY"] = api_key
        Settings.llm = Cohere(model=model, api_key=api_key)
    else:
        raise ValueError("Opción de LLM no válida. Usa 'ollama' o 'cohere'.")

    # Conexión a Chroma
    chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)

    if chroma_collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info(f"🔹 Usando índice existente con {chroma_collection.count()} vectores")
    else:
        docs = load_data_from_sql(engine, field_map=field_map)

        # Agregar esquema de tablas como contexto adicional
        schema_info = []
        for table in insp.get_table_names():
            cols = [col['name'] for col in insp.get_columns(table)]
            schema_info.append(f"{table}: {', '.join(cols)}")

        schema_doc = Document(
            text="Estructura de la base de datos:\n" + "\n".join(schema_info),
            metadata={"type": "schema"}
        )
        docs.append(schema_doc)

        if len(docs) == 0:
            raise ValueError("No se encontraron documentos para indexar.")

        index = VectorStoreIndex([], storage_context=storage_context)
        batch_insert(index, docs, batch_size=500)
        logger.info(f"🔹 Se creó un nuevo índice con {len(docs)} documentos")

    query_engine = index.as_query_engine(similarity_top_k=3)
    search_tool = make_search_tool(query_engine)

    system_prompt = f"""Eres un asistente especializado en responder preguntas sobre la base de datos SQL '{sql_uri}'.

Reglas:
1. Usa siempre la herramienta 'search_info' para buscar datos relevantes.
2. Explica los resultados de forma clara en español.
3. Incluye fuentes o tablas consultadas siempre que sea posible.
"""

    return FunctionCallingAgentWorker.from_tools(
        tools=[search_tool],
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True
    )


async def main():
    sql_uri = "mysql+pymysql://mi_usuario@localhost:3306/mi_base"

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

    agent = get_generic_sql_agent(
        sql_uri=sql_uri,
        chroma_collection_name="Inventario_de_emisiones",
        llm_choice="cohere",
        llm_params={
            "model":"command-r",
            "api_key":"djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"
            }
    ).as_agent()

    # Bucle de interacción
    print("🤖 Agente SQL listo. Escribe 'salir' para terminar.")
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
