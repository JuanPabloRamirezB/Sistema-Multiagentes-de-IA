# from fastapi import FastAPI
# from pydantic import BaseModel
# from air_agent import get_air_agent

# app = FastAPI()
# agent = get_air_agent().as_agent()

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/query")
# async def query_agent(req: QueryRequest):
#     response = await agent.achat(req.query)
#     return {"response": response.response}

# main.py (FastAPI endpoint)
from fastapi import FastAPI
from pydantic import BaseModel
from air_agent import get_air_agent
import logging

app = FastAPI()
agent = get_air_agent().as_agent() 

# Usar el mismo logger que en el agente para consistencia
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_agent(req: QueryRequest):
    try:
        logging.info(f"Consulta recibida: {req.query}") # 💡 Nuevo: Log de la consulta
        response = await agent.achat(req.query)
        print(f"response: {response.response}",flush=True)
        
        # 💡 Nuevo: Muestra el objeto de respuesta completo de LlamaIndex
        logging.info(f"Respuesta cruda del agente: {response}")

        if response and response.response:
            return {"response": response.response}
        else:
            return {"response": "No se obtuvo una respuesta válida del agente. Posible fallo del LLM al formular la respuesta final."}
    
    except Exception as e:
        logging.error(f"❌ ERROR AGENTE: {e}")
        return {"response": f"❌ Error al procesar la consulta: {e}"}