from pathlib import Path
import yaml
import os
import logging
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def train_model(config_file):
    '''
    FUnción principal que hace el entrenamiento del modelo.
    Args:
        config_file (str): Rutal del archivo de configuración con las variables del experimento
    '''
    # Cargar el archivo de configuración
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    target_variable = config['target_variable']
    columnas_config = config.get('columnas', [])
    # Lista de columnas establecidas en el archivo de configuración
    ignorar_cols = columnas_config.get('ignorar', [])
    num_cols = columnas_config.get('num', [])
    cat_ord_cols = columnas_config.get('cat_ord', [])
    cat_nom_ohe_drop_cols = columnas_config.get('cat_nom_ohe_drop', [])
    cat_nom_ohe_cols = columnas_config.get('cat_nom_ohe', [])
    cat_nom_frec_cols = columnas_config.get('cat_nom_frec', [])

    # Rutas del archivo de datos, ruta del modelo, metadata y lgs
    DATA_FILE = Path(os.getenv('DATA_FILE', BASE_DIR / 'data' / 'raw' / 'loan_dataset_20000.csv'))
    MODEL_DIR = Path(os.getenv('MODEL_DIR', BASE_DIR / 'models'))
    METADATA_DIR = Path(os.getenv('METADATA_DIR', BASE_DIR / 'metadata'))
    LOGS_DIR = Path(os.getenv('LOGS_DIR', BASE_DIR / 'logs'))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Configurar un archivo para hacer unos loggings
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(LOGS_DIR / config['log_file'])
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # Establecer las semillas para trazabilidad
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)

    # Cargar los datos para trabajarlos
    logger.info(f"Iniciando con el experimento {str(config['experiment_name'])}")
    if not DATA_FILE.exists():
        logger.error(f'Archivo no encontrado en: {DATA_FILE}')
        raise FileNotFoundError('No existen los datos')

    data = pd.read_csv(DATA_FILE)
    logger.info("Datos cargados correctamente")

    # Validación de los datos cargados
    if data.empty:
        logger.warning("El archivo de datos está vacío")
        raise ValueError("Datos vacíos")
    
    # Validar que las columnas son las esperadas que se van a usar
    required_cols = (
        [target_variable]+
        ignorar_cols+
        num_cols+
        cat_ord_cols+
        cat_nom_ohe_drop_cols+
        cat_nom_ohe_cols+ 
        cat_nom_frec_cols)
    
    faltantes = [col for col in required_cols if col not in data.columns]
    if faltantes:
        logger.error(f'Faltan columnas en los datos a trabajar: {faltantes}')
        raise ValueError(f'Faltan columnas: {faltantes}')
    logger.info("Todas las columnas están presentes")

    # Validación del conjunto de datos
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    val_config = config.get('data_split', [])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size= val_config.get('test_size', 0.2),
        random_state= seed,
        stratify= y)
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    pass

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_FILE = BASE_DIR / 'config' / '01_experimento.yaml'
    train_model(str(CONFIG_FILE))

