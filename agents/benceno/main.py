import os
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from benceno_agent import get_generic_mongo_agent

app = FastAPI()
# agent_worker = get_benceno_agent().as_agent()
# --- Parseo de argumentos ---
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
chroma_collection = os.getenv("CHROMA_COLLECTION")
llm_choice = os.getenv("LLM_CHOICE")
model = os.getenv("MODEL")
base_url_API_key = os.getenv("BASE_URL_API_KEY")

agent = get_generic_mongo_agent(
    mongo_uri=mongo_uri,
    db_name=db_name,
    chroma_collection_name = chroma_collection,
    field_map=None,
    llm_choice=llm_choice,  # o "ollama"
    llm_params={"model": model, 
                "base_url_API_key": base_url_API_key
                }
)
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_agent(req: QueryRequest):
    response = await agent.chat(req.query)
    return {"response": response.response}
