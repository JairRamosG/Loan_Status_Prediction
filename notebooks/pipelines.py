from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline

def build_preprocessor(columnas_config, preprocessor_config):
    """
    Construir los pipelines individuales para cada tipo de dato y despues unirlos en un 
    preprocesador con ColumnTransformer. Este es un elemento usado en build_full_pipeline.
    Args:
        columnas_config (dict): Parte correspondiente de las caracteristicas originales 
        preprocessor_config (dict): Parte correspondiente de variables del preprocesamiento 
    """

    # Columnas nuevas de ingenieria de características
    ignorar_cols = columnas_config.get('ignorar', [])
    num_cols = columnas_config.get('num_cols', [])
    cat_ord_cols = columnas_config.get('cat_ord_cols', [])
    cat_nom_ohe_drop_cols = columnas_config.get('cat_nom_ohe_drop', [])
    cat_nom_ohe_cols = columnas_config.get('cat_nom_ohe', [])
    cat_nom_ohe_frec_cols = columnas_config.get('cat_nom_frec', [])

    # Pipeline para datos numéricos
    num_cols_processing = Pipeline([
        ()
    ])

    # Pipeline para datos categóricos ordinales

    # Pipeline para datos categóricos nominales ohe
    pass

def build_model(models_config, seed):
    pass

def build_full_pipeline(config, seed):
    """
    COntruye el pipeline completo usando las funciones de preprocesamiento y la del modelo
    Args:
        config (dict): Es el archivo de configuración
        seed (int): Semilla para la trazabilidad
    """
    # Extraer información del archivo de configuración
    try:
        columnas_config = config.get('columnas', {})
        preprocessor_config = config.get('preprocessing', {})
        feature_engineering_config = config.get('feature_engineering', {})
        smote_config = config.get('SMOTE', {})
        models_config = config.get('models', {})
    except Exception as e:
        print(str(e)) 

    # Construcción de los componentes del pipeline
    preprocessor = build_preprocessor(columnas_config, preprocessor_config)
    model = build_model(models_config, seed)

    