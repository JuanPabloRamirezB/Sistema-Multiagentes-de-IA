import os
from fastapi import FastAPI
from pydantic import BaseModel
from mongo_agent import get_generic_mongo_agent

# --- Parseo de argumentos ---
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
chroma_collection = os.getenv("CHROMA_COLLECTION")
llm_choice = os.getenv("LLM_CHOICE")
model = os.getenv("MODEL")
base_url_API_key = os.getenv("BASE_URL_API_KEY")


# --- Configuración del agente ---
app = FastAPI()

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
    mongo_uri= mongo_uri,
    db_name= db_name,
    chroma_collection_name= chroma_collection,
    llm_choice = llm_choice,
        llm_params={
            "model": model,
            "base_url_API_key": base_url_API_key
        }
).as_agent()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_agent(req: QueryRequest):
    response = await agent.achat(req.query)
    return {"response": response.response}
