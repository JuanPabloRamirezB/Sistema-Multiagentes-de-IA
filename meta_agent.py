import httpx
import os
from typing import Optional, Dict
from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.agent.workflow import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.cohere import Cohere
from llama_index.llms.ollama import Ollama
import os

# 💡 Clase SafeOllama (manejo seguro de tool_calls)
class SafeOllama(Ollama):
    def get_tool_calls_from_response(self, *args, **kwargs):
        try:
            tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
        except TypeError:
            return []
        return tool_calls or []

async def call_agent(url: str, query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json={"query": query})
            response.raise_for_status()
            return response.json()["response"]
    except Exception as e:
        return f"[Error al contactar al agente: {str(e)}]"


# async def ask_benceno_agent(query: str) -> str:
#     return await call_agent("http://localhost:8002/query", query)

async def ask_air_quality(query: str) -> str:
    return await call_agent("http://api_externa_agent:8004/query", query)

async def ask_covid(query: str) -> str:
    return await call_agent("http://mongo_agent:8002/query", query)

async def ask_emisiones_contaminantes(query: str) -> str:
    return await call_agent("http://sql_agent:8003/query", query)

# Define el modelo compatible
# os.environ["COHERE_API_KEY"] = "djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"
# llm = Cohere(model="command-r", api_key=os.environ["COHERE_API_KEY"])

# --- Configuración dinámica del LLM ---
def configure_llm(llm_choice: str, llm_params: Optional[Dict[str, str]] = None):
    if llm_choice.lower() == "ollama":
        model = llm_params.get("model")
        base_url = llm_params.get("base_url")
        llm = SafeOllama(model=model, base_url=base_url, request_timeout=360.0)
    elif llm_choice.lower() == "cohere":
        model = llm_params.get("model")
        api_key = llm_params.get("api_key")
        os.environ["COHERE_API_KEY"] = api_key
        llm = Cohere(model=model, api_key=api_key)
    else:
        raise ValueError("❌ Opción de LLM no válida. Usa 'ollama' o 'cohere'.")
    return llm


# --- Meta-agente configurable ---
def get_meta_agent(
    llm_choice: str ,
    llm_params: Optional[Dict[str, str]] = None
):
    llm = configure_llm(llm_choice, llm_params)

    pollution_tool = FunctionTool.from_defaults(
    fn=ask_air_quality,
    name="ask_air_quality",
    description="Consulta sobre la calidad del aire en alguna ciudad"
    )
    covid_tool = FunctionTool.from_defaults(
    fn=ask_covid,
    name="ask_covid",
    description="Consulta sobre datos de covid"
    )
    covid_tool = FunctionTool.from_defaults(
    fn=ask_emisiones_contaminantes,
    name="ask_emisiones_contaminantes",
    description="Consulta sobre datos de emisiones"
    )
    return FunctionCallingAgentWorker.from_tools(
        tools=[pollution_tool,covid_tool],
        llm=llm,
        system_prompt="""Eres un meta-agente que delega consultas a otros agentes especializados. 
    Utiliza la herramienta adecuada según el tema de la pregunta."""
)

# import os
# import httpx
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.tools import FunctionTool
# from llama_index.llms.cohere import Cohere

# # 🧠 Memoria con ChromaDB
# from llama_index.core.memory import VectorMemory
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core.storage.chat_store import SimpleChatStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import chromadb

# # ---------- Funciones para llamar agentes ----------
# async def call_agent(url: str, query: str) -> str:
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             response = await client.post(url, json={"query": query})
#             response.raise_for_status()
#             return response.json()["response"]
#     except Exception as e:
#         return f"[Error al contactar al agente: {str(e)}]"

# async def ask_benceno_agent(query: str) -> str:
#     return await call_agent("http://localhost:8001/query", query)

# async def ask_air_quality(query: str) -> str:
#     return await call_agent("http://localhost:8003/query", query)


# # ---------- Configurar LLM ----------
# os.environ["COHERE_API_KEY"] = "djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"
# llm = Cohere(model="command-r", api_key=os.environ["COHERE_API_KEY"])


# # ---------- Configurar ChromaDB como almacenamiento de memoria ----------
# # Inicializa cliente Chroma
# CHROMA_HOST = "localhost"
# CHROMA_PORT = 8000
# COLLECTION_NAME_CHROMA = "meta_agent_memory"
# print(f"\n2. Conectando a ChromaDB en http://{CHROMA_HOST}:{CHROMA_PORT}...")
# try:
#     chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
#     print(f"Conexión a ChromaDB establecida. Usando colección: {COLLECTION_NAME_CHROMA}")
# except Exception as e:
#     print(f"Error al conectar a ChromaDB: {e}")
#     print("Asegúrate de que tu contenedor de ChromaDB esté corriendo. Ejecuta el 'docker run' apropiado.")
#     exit()



# # Embeddings para almacenar el contexto
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# # VectorStore usando Chroma
# vector_store = ChromaVectorStore(
#     chroma_collection=chroma_client.get_or_create_collection(COLLECTION_NAME_CHROMA)
# )

# # ChatStore para manejar los turnos de conversación
# chat_store = SimpleChatStore()

# # Memoria de conversación con embeddings
# # memory = ChatMemoryBuffer.from_defaults(
# #     chat_store=chat_store,
# #     vector_store=vector_store,
# #     embed_model=embed_model,
# #     token_limit=10000  # límite de contexto que guarda
# # )


# memory = VectorMemory(
#     # chat_store=chat_store,
#     vector_store=vector_store,
#     embed_model=embed_model,
#     # token_limit=10000
# )

# # ---------- Crear meta-agente con memoria ----------
# def get_meta_agent():
#     benceno_tool = FunctionTool.from_defaults(
#         fn=ask_benceno_agent,
#         name="ask_benceno_agent",
#         description="Consulta sobre emisiones de benceno"
#     )
#     pollution_tool = FunctionTool.from_defaults(
#         fn=ask_air_quality,
#         name="ask_air_quality",
#         description="Consulta sobre la calidad del aire en alguna ciudad"
#     )

#     return FunctionCallingAgentWorker.from_tools(
#         tools=[benceno_tool, pollution_tool],
#         llm=llm,
#         memory=memory,  # 🧠 Memoria persistente
#         system_prompt="""Eres un meta-agente que delega consultas a otros agentes especializados. 
#         Utiliza la herramienta adecuada según el tema de la pregunta."""
#     )

