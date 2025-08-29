# Sistema multiagente de IA (Atahualpa)

Este proyecto implementa un **agente de LlamaIndex** que utiliza **MongoDB** como base de datos.  
El agente se expone mediante una **API con FastAPI**, lista para recibir consultas de los usuarios.

---

## 🚀 Requisitos

- Python 3.10+
- Docker (si quieres ejecutar MongoDB y ChromaDB en contenedores)
- Una base de datos MongoDB poblada con la colección
- Variables de entorno configuradas

---

## Despliegue

Para desplegar todo el sistema se puede utilizar como base el compose de prueba y organizarlo a los usos especificos.

El compose se compone por las bases de datos que se requiriron para la experimentacion, sin embargo no son necesarios todos, el cliente de Ollama y los Agentes.

### Bases de Datos

- Se emplea **ChromaDB** como una base de datos vectorial que funge como CAG (Indispensable para todos los agentes).

- Se emplea **MongoDB** como una fuente de datos (RAG).(Opcional, pero la mas estable)

- Se emplea **MySql** Como una fuente de datos (RAG). (Opcional, pero no fue probada tan exaustivamente).

Yo recomiendo que unicamente dejes ChromaDB como se encuentra y utilices Mongo u MySql segun el caso de uso.

### Ollama

- Se emplea el cliente de **Ollama** el cual es el que contiene el LLM **Llama3.2** para ejecutar de manera autoalojada. (Recomendado para ejecutar los agentes. Si se desea cambiar de modelo debes editar las lineas 14 y 17 del archivo **entrypoint.sh** antes de iniciar el compose.)

### Agentes de IA
Los agentes se dividen en dos que trabajan de forma estable, los que son para Mongo y los que son para SQL, ambos requieren de variables de entorno configurables en el compose.
**mongo_agent**:
      - MONGO_URI=mongodb://mongodb:27017/ (Que es la conexion al contenedor de mongo)
      - DB_NAME=covid_db    (El nombre de la base de datos donde se almacenan)
      - CHROMA_COLLECTION=casos_covid   (El nombre de la coleccion dentro de ChromaDB)

**LLM_CHOICE** Para estas variables se tiene dos opciones. las que son con ollama o con un LLM externa(Por el momento solo Cohere)

        Opcion1
      - LLM_CHOICE=cohere
      - MODEL=command-r
      - BASE_URL_API_KEY=djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB

        Opcion2
      - LLM_CHOICE=ollama
      - MODEL=llama3.2
      - BASE_URL_API_KEY=http://ollama:11434

**sql_agent**:
      - SQL_URI=mysql+pymysql://mi_usuario@mysql:3306/mi_base (Que es la conexion al contenedor de MySql)
      - CHROMA_COLLECTION=Inventario_de_emisiones   (El nombre de la coleccion dentro de ChromaDB)

**LLM_CHOICE** Para estas variables se tiene dos opciones. las que son con ollama o con un LLM externa(Por el momento solo Cohere)

        Opcion1
      - LLM_CHOICE=cohere
      - MODEL=command-r
      - BASE_URL_API_KEY=djpip5AznRIul3xtChGzwq92kRuNHlf9lLjctBQB

        Opcion2
      - LLM_CHOICE=ollama
      - MODEL=llama3.2
      - BASE_URL_API_KEY=http://ollama:11434

- El agente de API externa aun no funciona de manera correcta, esta hardcode, asi que no recomiendo usarlo.

- El meta agente aun sigue en desarrollo.

