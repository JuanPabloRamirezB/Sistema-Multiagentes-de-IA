import pandas as pd
from sqlalchemy import create_engine
import polars as pl

def upload_csv_to_mysql(
    csv_path: str,
    mysql_uri: str = "mysql+pymysql://usuario:contraseña@localhost:3306/",
    db_name: str = "mi_base",
    table_name: str = "mi_tabla",
    if_exists: str = "append",
    chunksize: int = 1000
):
    """
    Sube un CSV a MySQL usando Polars y SQLAlchemy.

    Args:
        csv_path (str): Ruta del archivo CSV.
        mysql_uri (str): URI de conexión a MySQL (formato: mysql+pymysql://usuario:contraseña@host:puerto/)
        db_name (str): Nombre de la base de datos.
        table_name (str): Nombre de la tabla.
        if_exists (str): Qué hacer si la tabla existe ('fail', 'replace', 'append').
        chunksize (int): Número de filas por lote al insertar (evita timeouts).
    """
    # Crear conexión con SQLAlchemy
    engine = create_engine(f"{mysql_uri}{db_name}")

    # Leer CSV con Polars
    df = pl.read_csv(csv_path)

    # Ejemplo de filtrado opcional
    # df = df.filter(pl.col("columna") > 10)

    # Convertir a diccionarios para insertar por lotes
    records = df.to_dicts()

    if not records:
        print("⚠️ No se encontraron datos en el CSV.")
        return

    # Insertar en lotes para evitar pérdida de conexión
    with engine.begin() as conn:
        for i in range(0, len(records), chunksize):
            chunk = records[i:i + chunksize]
            pl.DataFrame(chunk).to_pandas().to_sql(
                table_name, con=conn, if_exists=if_exists, index=False
            )
            print(f"✅ Insertadas {len(chunk)} filas en {db_name}.{table_name}")

    print(f"🎯 Total insertado: {len(records)} filas en '{db_name}.{table_name}'")

if __name__ == "__main__":
    upload_csv_to_mysql(
        csv_path="d3_aire01_49_1.csv",  
        mysql_uri="mysql+pymysql://mi_usuario@localhost:3306/",
        db_name="mi_base",
        table_name="Inventario_de_emisiones_de_contaminantes",
        if_exists="append"
    )
