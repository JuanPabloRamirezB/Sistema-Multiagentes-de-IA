import argparse
import os
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from sql_agent import get_generic_sql_agent

# --- Parseo de argumentos y Configuración de variables de entorno ---
# sql_uri = "mysql+pymysql://mi_usuario@localhost:3306/mi_base"
sql_uri = os.getenv("SQL_URI")
chroma_collection = os.getenv("CHROMA_COLLECTION")
llm_choice = os.getenv("LLM_CHOICE")
model = os.getenv("MODEL")
base_url_API_key = os.getenv("BASE_URL_API_KEY")

# --- Configuración del agente ---
app = FastAPI()
agent = get_generic_sql_agent(
    sql_uri=sql_uri,
    chroma_collection_name=chroma_collection,
    llm_choice="cohere",
    llm_params={
        "model":"command-r",
        "api_key":"djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB"
        }
).as_agent()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_agent(req: QueryRequest):
    response = await agent.achat(req.query)
    return {"response": response.response}