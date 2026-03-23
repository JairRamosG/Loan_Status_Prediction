from pathlib import Path
import yaml
import os
import logging
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from pipelines import build_full_pipeline
from datetime import datetime

from utils import save_learning_curve, save_confusion_matrix, save_medidas_biclase
import joblib
import json


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
    NUM_COLS = columnas_config.get('num_cols', [])
    CAT_ORD_COLS = columnas_config.get('cat_ord_cols', [])
    CAT_NOM_OHE_DROP_COLS = columnas_config.get('cat_nom_ohe_drop', [])
    CAT_NOM_OHE_COLS = columnas_config.get('cat_nom_ohe', [])
    CAT_NOM_OHE_FREC_COLS = columnas_config.get('cat_nom_frec', [])

    # Rutas del archivo de datos, ruta del modelo, metadata y lgs
    DATA_FILE = Path(os.getenv('DATA_FILE', BASE_DIR / 'data' / 'raw' / 'loan_dataset_20000.csv'))
    MODEL_DIR = Path(os.getenv('MODEL_DIR', BASE_DIR / 'models'))
    METADATA_DIR = Path(os.getenv('METADATA_DIR', BASE_DIR / 'metadata'))
    LOGS_DIR = Path(os.getenv('LOGS_DIR', BASE_DIR / 'logs'))
    LOGS_FILIE_PATH = LOGS_DIR / f"{config['experiment_name']}.log"
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"LOGS_DIR: {LOGS_DIR}")
    print(f"Log file: {LOGS_DIR / config['experiment_name']}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Configurar un archivo para hacer unos loggings
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(LOGS_FILIE_PATH)
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
        list(NUM_COLS.keys())+
        list(CAT_ORD_COLS.keys())+
        CAT_NOM_OHE_DROP_COLS+
        CAT_NOM_OHE_COLS+
        CAT_NOM_OHE_FREC_COLS)
    faltantes = [col for col in required_cols if col not in data.columns]
    if faltantes:
        logger.error(f'Faltan columnas: {faltantes}')
        raise ValueError(f'Flatan columnas: {faltantes}')
    
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
    logger.info(f"\tX_train: {X_train.shape}")
    logger.info(f"\tX_test : {X_test.shape}")
    logger.info(f"\ty_train: {y_train.shape}")
    logger.info(f"\ty_test : {y_test.shape}")

    # Identificacion de las columnas
    ignorar_cols = IGNORAR_COLS.copy()
    num_cols = NUM_COLS.copy()
    cat_ord_cols = CAT_ORD_COLS.copy()
    cat_nom_ohe_drop_cols = CAT_NOM_OHE_DROP_COLS.copy()
    cat_nom_ohe_cols = CAT_NOM_OHE_COLS.copy()
    cat_nom_ohe_frec_cols = CAT_NOM_OHE_FREC_COLS.copy()
    
    logger.info("Columnas cargadas del archivo de configuración:")
    logger.info(f"\t{len(ignorar_cols)}\t Ignoradas") 
    logger.info(f"\t{len(num_cols)}\t Numericas") 
    logger.info(f"\t{len(cat_ord_cols)}\t Categoricas ordinales")
    logger.info(f"\t{len(cat_nom_ohe_drop_cols)}\t Categoricas nominales ohe drop")
    logger.info(f"\t{len(cat_nom_ohe_cols)}\t Categoricas nominales ohe")
    logger.info(f"\t{len(cat_nom_ohe_frec_cols)}\t Categoricas nominales freq ohe")

    # COnstrucción del pipeline
    try:
        pipeline = build_full_pipeline(config, seed)
        logger.info('Pipeline construido exitosamente')
        logger.info(f'Pasos: {list(pipeline.named_steps.keys())}')
    except Exception as e:
        logger.error(f'Error contruyendo el pipeline: {str(e)}')
        raise

    # Obtener la configuración para la validación cruzada
    grid_config = config.get('random_search', {})

    tipo = grid_config.get('tipo')
    n_iter = grid_config.get('n_iter', 25)
    scoring = grid_config.get('scoring', 'precision')
    cv_folds = grid_config.get('cv_folds', 3)
    n_jobs = grid_config.get('n_jobs', -1)
    verbose = grid_config.get('verbose', 1)
    error_score = grid_config.get('error_score', 'raise')
    param_grid = grid_config.get('param_grid', {})

    logger.info('Configuración de CV:')
    logger.info(f'   Tipo: {tipo}')
    logger.info(f'   n_iter: {n_iter}')
    logger.info(f'   scoring: {scoring}')
    logger.info(f'   cv_folds: {cv_folds}')
    logger.info(f'   n_jobs: {n_jobs}')
    logger.info(f'   verbose: {verbose}')
    logger.info(f'   error_score: {error_score}')

    # Construir el RandomizedSearchCV
    grid = RandomizedSearchCV(
        estimator = pipeline,
        param_distributions = param_grid,
        n_iter = n_iter,
        scoring = scoring,
        cv = cv_folds,
        n_jobs = n_jobs,
        verbose = verbose,
        error_score = error_score,
        random_state = seed
        )
    
    # Entrenamiento
    start_time = datetime.now()
    try:
        logger.info('Inicio del entrenamiento')
        grid.fit(X_train, y_train)
        total_time = datetime.now() - start_time
        logger.info(f'Final del entrenamiento en: {total_time}')    
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        raise

    # Resultados del entrenamiento
    try:
        logger.info(f'Mejores parámetros:')
        for param, value in grid.best_params_.items():
            logger.info(f'          {param}: {value}')            
        logger.info(f'Mejor score:  {np.round(grid.best_score_, 4)}')
        best_model = grid.best_estimator_
        logger.info('Mejor modelo obtenido del RandomizedSearchCV')
    except Exception as e:
        logger.error(f'Error en los resultados del grid: {str(e)}')
        raise

    # Curva de aprendizaje
    try:
        ruta_curva = str(METADATA_DIR) + "/" + config['experiment_name'] + '_learning_curve' + '.png'
        save_learning_curve(
        estimator=best_model,
        X=X_train,
        y=y_train,
        ruta_img = ruta_curva,
        scoring = scoring,    
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
        title=f"Curva de Aprendizaje - {config['experiment_name']} (mejor modelo)"
    )
    except Exception as e:
        logger.error(f'Error generando la curva: {str(e)}', exc_info=True)

    # Evaluación del modelo
    try:
        y_pred = best_model.predict(X_test)
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test)[:, 1]
            logger.info('Probabilidades obtenidas correctamente')        
    except Exception as e:
        logger.error(f'Falla en la evaluacion del modelo {str(e)}')
        raise

    # REGISTRAR LA MATRIZ
    try:
        ruta_img = str(METADATA_DIR) + "/" + config['experiment_name'] + '_matriz' + '.png'
        save_confusion_matrix(y_test, y_pred, ruta_img)
        logger.info('Matriz de confusión guardada exitosamente')
    except Exception as e:
        logger.error(f'Falla en el registro de la matriz {str(e)}')
        raise

    # REGISTRAR LAS MEDIDAS
    try:
        ruta_medidas = str(METADATA_DIR) + "/" + config['experiment_name'] + '_medidas' + '.csv'
        medidas = save_medidas_biclase(y_test, y_pred, ruta_medidas)
        logger.info('Medidas de desempeño guardadas exitosamente')
    except Exception as e:
        logger.error(f'Falla en el registro de las medidas de desempeño{str(e)}')
        raise

    # REGISTRAR LOS METADATOS EN UN JSON
    try:
        ruta_metadatos = str(METADATA_DIR) + "/" + config['experiment_name'] + '.json'
        metadata = {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "scoring": medidas.to_dict()}
        with open(ruta_metadatos, 'w') as f:
            json.dump(metadata, f, indent = 4)
            logger.info('Metadatos registrados correctamente')
    except Exception as e:
        logger.error(f'Error durante el registro de los metadatos: {str(e)}')
        raise

    # REGISTRAR EL MODELO
    try:
        ruta_modelo = str(MODEL_DIR) + "/" + config['experiment_name'] + '.pkl'
        joblib.dump(best_model, str(ruta_modelo))
        logger.info(f"Modelo {str(config['experiment_name'])} guardado exitosamente")
    except Exception as e:
        logger.error(f'Falla guardando el modelo: {str(e)}')
        raise

    logger.info('Metadatos registrados exitosamente')
    print('=='*50)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_FILE = BASE_DIR / 'config' / '02_experimento.yaml'
    train_model(str(CONFIG_FILE))

