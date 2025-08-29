import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from meta_agent import get_meta_agent

llm_choice = os.getenv("LLM_CHOICE")
model = os.getenv("MODEL")
base_url_API_key = os.getenv("BASE_URL_API_KEY")

# --- Configuración del agente ---
app = FastAPI()
agent = get_meta_agent(
    llm_choice="cohere",
    llm_params={
        "model":"command-r",
        "api_key":"tEoaKGsnrDb7iS4HKF1v1dHAhPfYBXRSpzgEg8K2"
        }
).as_agent()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_agent(req: QueryRequest):
    response = await agent.achat(req.query)
    return {"response": response.response}
