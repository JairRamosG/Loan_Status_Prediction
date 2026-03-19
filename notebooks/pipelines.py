from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from utils import FeatureEngineering
from imblearn.over_sampling import SMOTE

def build_preprocessor(columnas_config, preprocessor_config):
    """
    Construir los pipelines individuales para cada tipo de dato y despues unirlos en un 
    preprocesador con ColumnTransformer. Este es un elemento usado en build_full_pipeline.
    Args:
        columnas_config (dict): Parte correspondiente de las caracteristicas originales 
        preprocessor_config (dict): Parte correspondiente de variables del preprocesamiento 
    """

    # Columnas nuevas de ingenieria de características
    ignorar = columnas_config.get('ignorar', {})
    num_dict = columnas_config.get('numericas', {})
    ord_dict = columnas_config.get('ordinales', {})
    nom_drop_list = columnas_config.get('nominales_ohe_drop', [])
    nom_ohe_list = columnas_config.get('nominales_ohe', [])
    nom_frec_list = columnas_config.get('nominales_frecuencia', [])

    # Pipeline para datos numéricos

    # Pipeline para datos categóricos ordinales

    # Pipeline para datos categóricos nominales ohe
    pass

build_preprocessor()

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

    # Insertar las variables nuevas de Feature Engineering dinamicamente
    if feature_engineering_config.get('create_age_group', False):
        columnas_config['cat_ord'] = columnas_config.get('cat_ord', []) + ['age_group']
    
    if feature_engineering_config.get('create_loan_to_income', False):
        columnas_config['numeric_cols'] = columnas_config.get('numeric_cols', []) + ['loan_to_income']

    if feature_engineering_config.get('create_has_delinquency_story', False):
        columnas_config['numeric_cols'] = columnas_config.get('numeric_cols', []) + ['has_delinquency_story']
    
    if feature_engineering_config.get('create_severity_score', False):
        columnas_config['numeric_cols'] = columnas_config.get('numeric_cols', []) + ['severity_score']

    if feature_engineering_config.get('create_payment_income', False):
        columnas_config['numeric_cols'] = columnas_config.get('numeric_cols', []) + ['payment_income']

    # Construcción de los componentes del pipeline
    preprocessor = build_preprocessor(columnas_config, preprocessor_config)
    model = build_model(models_config, seed)

    # Ensablo el pipeline para tratar los datos y entrenar el modelo
    pipeline = Pipeline([
        ('feature_engineering', FeatureEngineering(**feature_engineering_config)),
        ('preprocessor', preprocessor),
        ('smote', SMOTE(**smote_config)),
        ('model', model)
    ])

    return pipeline