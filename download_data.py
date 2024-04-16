import subprocess
import zipfile
import os
from dotenv import load_dotenv
load_dotenv()

def unzip(archivo_zip, carpeta_destino):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(carpeta_destino)

    os.remove(archivo_zip)

def descargar_dataset_kaggle(dataset_url):
    """
    Descarga un dataset de Kaggle utilizando el comando 'kaggle datasets download'.
    
    Args:
        dataset_url (str): URL del dataset de Kaggle a descargar.
    """
    cmd = f"kaggle datasets download -d {dataset_url}"
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    descargar_dataset_kaggle("paoloripamonti/derma-diseases")
    unzip("derma-diseases.zip", "data")