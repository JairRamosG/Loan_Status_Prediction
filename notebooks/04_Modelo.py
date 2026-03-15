from pathlib import Path
import yaml
import os
import logging

def train_model(config_file):
    '''
    FUnción principal que hace el entrenamiento del modelo.
    Args:
        config_file (str): Rutal del archivo de configuración con las variables del experimento
    '''
    # Cargar el archivo de configuración
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
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

    print(type(logger))

    pass

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_FILE = BASE_DIR / 'config' / '01_experimento.yaml'
    train_model(str(CONFIG_FILE))

