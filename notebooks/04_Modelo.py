from pathlib import Path
import yaml

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

    print(num_cols)
    pass

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_FILE = BASE_DIR / 'config' / '01_experimento.yaml'
    print(CONFIG_FILE)
    train_model(str(CONFIG_FILE))

