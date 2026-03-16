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
    
    TARGET_VARIABLE = config['target_variable']
    columnas_config = config.get('columnas', [])
    # Lista de columnas establecidas en el archivo de configuración
    IGNORAR_COLS = columnas_config.get('ignorar', [])
    NUM_COLS = columnas_config.get('num', [])
    CAT_ORD_COLS = columnas_config.get('cat_ord', [])
    CAT_NOM_OHE_DROP_COLS = columnas_config.get('cat_nom_ohe_drop', [])
    CAT_NOM_OHE_COLS = columnas_config.get('cat_nom_ohe', [])
    CAT_NOM_OHE_FREC_COLS = columnas_config.get('cat_nom_frec', [])

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
        [TARGET_VARIABLE]+
        IGNORAR_COLS+
        NUM_COLS+
        CAT_ORD_COLS+
        CAT_NOM_OHE_DROP_COLS+
        CAT_NOM_OHE_COLS+
        CAT_NOM_OHE_FREC_COLS)
    
    faltantes = [col for col in required_cols if col not in data.columns]
    if faltantes:
        logger.error(f'Faltan columnas en los datos a trabajar: {faltantes}')
        raise ValueError(f'Faltan columnas: {faltantes}')
    logger.info("Todas las columnas están presentes")

    # Validación del conjunto de datos
    X = data.drop(columns=[TARGET_VARIABLE])
    y = data[TARGET_VARIABLE]

    val_config = config.get('data_split', [])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size= val_config.get('test_size', 0.2),
        random_state= seed,
        stratify= y)
    logger.info("Validación del conjunto de datos realizada")
    logger.info(f" X_train: {X_train.shape}")
    logger.info(f" X_test : {X_test.shape}")
    logger.info(f" y_train: {y_train.shape}")
    logger.info(f" y_test : {y_test.shape}")

    # Identificacion de las columnas
    ignorar_cols = IGNORAR_COLS.copy()
    num_cols = NUM_COLS.copy()
    cat_ord_cols = CAT_ORD_COLS.copy()
    cat_nom_ohe_drop_cols = CAT_NOM_OHE_DROP_COLS.copy()
    cat_nom_ohe_cols = CAT_NOM_OHE_COLS.copy()
    cat_nom_ohe_frec_cols = CAT_NOM_OHE_FREC_COLS.copy()
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    pass

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_FILE = BASE_DIR / 'config' / '01_experimento.yaml'
    train_model(str(CONFIG_FILE))

