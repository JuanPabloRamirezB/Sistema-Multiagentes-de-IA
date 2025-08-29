import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from generic_agent import get_generic_mongo_agent

# --- Parseo de argumentos ---
# parser = argparse.ArgumentParser()
# parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
# parser.add_argument("--db-name", default="covid_db")
# parser.add_argument("--chroma-collection", default="casos_covid")
# args, _ = parser.parse_known_args()

# --- Configuración del agente ---
app = FastAPI()
agent = get_generic_mongo_agent(
    mongo_uri="mongodb://localhost:27017/",#args.mongo_uri,
    db_name="covid_db",#args.db_name,
    chroma_collection_name="casos_covid",#args.chroma_collection
).as_agent()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_agent(req: QueryRequest):
    response = await agent.achat(req.query)
    return {"response": response.response}
