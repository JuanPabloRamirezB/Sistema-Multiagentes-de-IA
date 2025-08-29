import pandas as pd
from pymongo import MongoClient

def upload_csv_to_mongo(
    csv_path: str,
    mongo_uri: str = "mongodb://localhost:27017/",
    db_name: str = "mi_base",
    collection_name: str = "mi_coleccion"
):
    """
    Sube un CSV a MongoDB.
    
    Args:
        csv_path (str): Ruta del archivo CSV.
        mongo_uri (str): URI de conexión a MongoDB.
        db_name (str): Nombre de la base de datos.
        collection_name (str): Nombre de la colección.
    """
    # Conectar a MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Leer CSV con pandas (detecta delimitador automáticamente)
    df = pd.read_csv(csv_path)

    # Rellenar NaN con None (Mongo no acepta NaN)
    df = df.where(pd.notnull(df), None)

    # Convertir a diccionarios
    data = df.to_dict(orient="records")

    # Insertar en Mongo
    if data:
        collection.insert_many(data)
        print(f"✅ Se insertaron {len(data)} documentos en '{db_name}.{collection_name}'")
    else:
        print("⚠️ No se encontraron datos en el CSV.")

if __name__ == "__main__":
    # Ejemplo con tu archivo COVID
    upload_csv_to_mongo(
        csv_path="covid_19_data.csv",          # Cambia por tu archivo
        mongo_uri="mongodb://localhost:27017/",
        db_name="covid_db",
        collection_name="casos_covid"
    )
