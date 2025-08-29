# import os
# import httpx
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.tools import FunctionTool
# from llama_index.llms.ollama import Ollama
# from llama_index.llms.cohere import Cohere


# OPENWEATHER_API_KEY = "5dd416e4c2d99962698c94584d102346"



# async def get_air_quality(city: str) -> str:
#     url = (
#     f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
#     )

#     async with httpx.AsyncClient() as client:
#         geo_response = await client.get(url)
#         geo_data = geo_response.json()

#     if not geo_data:
#         return f"No se encontró la ciudad '{city}'."

#     lat = geo_data[0]["lat"]
#     lon = geo_data[0]["lon"]

#     url = (
#         f"http://api.openweathermap.org/data/2.5/air_pollution"
#         f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
#     )
#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         data = response.json()

#     if "list" not in data:
#         return "No se pudo obtener la calidad del aire."

#     aqi_data = data["list"][0]
#     components = aqi_data["components"]

#     return (
#         f"Calidad del aire en {city.title()}:\n"
#         f"- PM2.5: {components['pm2_5']} µg/m³\n"
#         f"- PM10: {components['pm10']} µg/m³\n"
#         f"- NO2: {components['no2']} µg/m³\n"
#         f"- O3: {components['o3']} µg/m³\n"
#         f"- CO: {components['co']} µg/m³\n"
#         f"- SO2: {components['so2']} µg/m³"
#     )

# tool = FunctionTool.from_defaults(
#     fn=get_air_quality,
#     name="get_air_quality",
#     description="Consulta la calidad del aire por ciudad usando OpenWeatherMap"
# )

# # os.environ["COHERE_API_KEY"] = "djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB" 
# # llm = Cohere(model="command-r", api_key=os.environ["COHERE_API_KEY"])
# # llm = Ollama(model="llama3.2", request_timeout=360.0)
# llm = Ollama(
#     model="llama3.2",
#     request_timeout=360.0,
#     base_url="http://localhost:11434"
# )

# def get_air_agent():
#     return FunctionCallingAgentWorker.from_tools(
#         tools=[tool],
#         llm=llm,
#         system_prompt="""
# Eres un asistente especializado en calidad del aire. Puedes obtener datos de contaminación por ciudad usando una API pública.
# Usa la herramienta 'get_air_quality' cuando se pregunte por niveles de PM2.5, PM10, NO2, etc.
# """
#     )
###################################################################################################
# air_agent.py
import os
import httpx
import json
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # Cambia a DEBUG para ver más detalles
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))    

OPENWEATHER_API_KEY = "5dd416e4c2d99962698c94584d102346"

# ===============================
# Función que obtiene calidad del aire
# ===============================
async def get_air_quality(city: str) -> str:
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    async with httpx.AsyncClient() as client:
        geo_response = await client.get(geo_url)
        geo_data = geo_response.json()

    if not geo_data:
        return f"No se encontró la ciudad '{city}'."

    lat = geo_data[0]["lat"]
    lon = geo_data[0]["lon"]

    air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(air_url)
        data = response.json()

    if "list" not in data:
        return "No se pudo obtener la calidad del aire."

    components = data["list"][0]["components"]

    return (
        f"Calidad del aire en {city.title()}:\n"
        f"- PM2.5: {components['pm2_5']} µg/m³\n"
        f"- PM10: {components['pm10']} µg/m³\n"
        f"- NO2: {components['no2']} µg/m³\n"
        f"- O3: {components['o3']} µg/m³\n"
        f"- CO: {components['co']} µg/m³\n"
        f"- SO2: {components['so2']} µg/m³"
    )

# ===============================
# Tool
# ===============================
tool = FunctionTool.from_defaults(
    fn=get_air_quality,
    name="get_air_quality",
    description="Consulta la calidad del aire por ciudad usando OpenWeatherMap"
)

# ===============================
# LLM
# ===============================
llm = Ollama(
    model="gpt-oss:20b",
    request_timeout=360.0,
    base_url="http://localhost:11434"
)

# ===============================
# Patch para tool_calls 
# ===============================
original_get_tool_calls = Ollama.get_tool_calls_from_response
def safe_get_tool_calls(self, *args, **kwargs):
    res = original_get_tool_calls(self, *args, **kwargs)
    return res or []
Ollama.get_tool_calls_from_response = safe_get_tool_calls

# ===============================
# Agente
# ===============================
def get_air_agent():
    return FunctionCallingAgentWorker.from_tools(
        tools=[tool],
        llm=llm,
        system_prompt="""
Eres un asistente especializado en calidad del aire. Utiliza la herramienta 'get_air_quality' para obtener
datos de calidad del aire para una ciudad específica.

Cuando un usuario pregunte por la calidad del aire, usa la herramienta. Después de obtener la respuesta,
formatea los resultados en un resumen claro y amigable para el usuario. Si la pregunta no es sobre la
calidad del aire, responde directamente de manera útil y concisa.
"""
    )
