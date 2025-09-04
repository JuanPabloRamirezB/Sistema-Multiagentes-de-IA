import os
from fastapi import FastAPI
from pydantic import BaseModel
from agent_api_externa import get_generic_waqi_agent



llm_choice = os.getenv("LLM_CHOICE")
model = os.getenv("MODEL")
base_url_API_key = os.getenv("BASE_URL_API_KEY")

app = FastAPI()
# Aquí eliges el LLM dinámicamente
agent = get_generic_waqi_agent(
    llm_choice=llm_choice,
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
