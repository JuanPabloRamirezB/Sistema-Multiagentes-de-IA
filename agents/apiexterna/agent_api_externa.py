import os
import asyncio
import httpx
import json
import logging
import sys
import traceback
from typing import Optional, Dict
from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.agent.workflow import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.cohere import Cohere
from llama_index.llms.ollama import Ollama

# Configurar logging
logger = logging.getLogger(__name__)

# 💡 Clase SafeOllama (manejo seguro de tool_calls)
class SafeOllama(Ollama):
    def get_tool_calls_from_response(self, *args, **kwargs):
        try:
            tool_calls = super().get_tool_calls_from_response(*args, **kwargs)
        except TypeError:
            return []
        return tool_calls or []

# === Herramienta WAQI ===
async def get_air_quality_waqi(city: str) -> str:
    """Obtiene la calidad del aire para una ciudad usando la API de WAQI."""
    token = os.getenv("WAQI_TOKEN", "098caad81ced20fc3b8d994841c77b48197c4335")
    if not token:
        raise ValueError("Debes definir la variable de entorno WAQI_TOKEN")
    
    url = f"http://api.waqi.info/feed/{city}/?token={token}"
    logger.info(f"Consultando WAQI para ciudad: {city}")
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=20.0)
            resp.raise_for_status()
            
            data = resp.json()
            logger.info(f"Respuesta WAQI: {data}")
            
            if data.get('status') == 'ok':
                aqi_data = data.get('data', {})
                city_name = aqi_data.get('city', {}).get('name', city)
                aqi_value = aqi_data.get('aqi', 'N/A')
                
                result = {
                    "ciudad": city_name,
                    "aqi": aqi_value,
                    "status": "ok",
                    "raw_data": data
                }
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return json.dumps({
                    "error": f"Error en API WAQI: {data.get('data', 'Error desconocido')}",
                    "status": "error"
                })
                
    except Exception as e:
        return json.dumps({"error": f"Error al consultar WAQI: {str(e)}", "status": "error"})

# === Construir agente genérico WAQI ===
def get_generic_waqi_agent(
    llm_choice: str,
    llm_params: Optional[Dict[str, str]] = None
):
    waqi_tool = FunctionTool.from_defaults(
        fn=get_air_quality_waqi,
        name="get_air_quality_waqi",
        description="Obtiene la calidad del aire para una ciudad específica usando la API WAQI."
    )

    # === Selección dinámica del LLM ===
    if llm_choice.lower() == "ollama":
        model = llm_params.get("model")
        base_url = llm_params.get("base_url_API_key")
        llm = SafeOllama(model=model, base_url=base_url, request_timeout=360.0)
        logger.info("✅ Usando SafeOllama como LLM principal")
    elif llm_choice.lower() == "cohere":
        model = llm_params.get("model")
        api_key = llm_params.get("base_url_API_key")
        os.environ["COHERE_API_KEY"] = api_key
        llm = Cohere(model=model, api_key=api_key)
        logger.info("✅ Usando Cohere como LLM principal")
    else:
        raise ValueError("❌ Opción de LLM no válida. Usa 'ollama' o 'cohere'.")

    system_prompt = """Eres un asistente especializado en consultas de calidad del aire.

Cuando un usuario pregunte sobre la calidad del aire de una ciudad:
1. Usa la herramienta 'get_air_quality_waqi' con el nombre de la ciudad
2. Interpreta los datos JSON devueltos
3. Proporciona una respuesta clara y útil en español
4. Responde mostrando un resumen del JSON crudo devuelto por la API.
"""
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=[waqi_tool],
        llm=llm,
        system_prompt=system_prompt,
        verbose=True
    )
    
    return agent_worker


# ---
## Main asincrónico
async def main():
    agent = get_generic_waqi_agent().as_agent()
    print("🌍 Agente WAQI listo. Escribe 'salir' para terminar.")

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

        except Exception as e:
            print(f"❌ Error: {e}")
            print("\nDetalles del error (Traceback):")
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())

