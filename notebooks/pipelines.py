from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from utils import FeatureEngineering
import importlib
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder

def build_preprocessor(columnas_config, preprocessor_config):
    """
    Construir dinamicamente los pipelines individuales para cada tipo de dato y despues unirlos en un 
    preprocesador con ColumnTransformer. Este es un elemento usado en build_full_pipeline.
    Args:
        columnas_config (dict): Parte correspondiente de las caracteristicas originales 
        preprocessor_config (dict): Parte correspondiente de variables del preprocesamiento 
    """

    # Columnas nuevas de ingenieria de características
    num_cols = columnas_config.get('num_cols', {})
    cat_ord_cols = columnas_config.get('cat_ord_cols', {})
    cat_nom_ohe_drop = columnas_config.get('cat_nom_ohe_drop', [])
    cat_nom_ohe = columnas_config.get('cat_nom_ohe', [])
    #cat_nom_frec = columnas_config.get('cat_nom_frec', [])

    transformers = []

    # Pipeline para datos numéricos
    transform_groups = {}
    for col, conf in num_cols.items():
        t = conf.get('transform', 'passthrough')
        transform_groups.setdefault(t, []).append(col)
    
    for t, cols in transform_groups.items():
        if t == 'passthrough':
            transformers.append((f'num_passthrough_{len(transformers)}', 'passthrough', cols))

        elif t == 'scale':
            pipeline = Pipeline([('scaler', StandardScaler())])
            transformers.append((f'scale_{len(transformers)}', pipeline, cols))

        elif t == 'log_scale':
            pipeline = Pipeline([
                ('log', FunctionTransformer(np.log, validate=True)),
                ('scaler', StandardScaler())
                ])
            transformers.append((f'log_scale_{len(transformers)}', pipeline, cols))

        elif t == 'log1p_scale':
            pipeline = Pipeline([
                ('log1p', FunctionTransformer(np.log1p, validate= True)),
                ('scaler', StandardScaler())])
            transformers.append((f'log1p_scale_{len(transformers)}', pipeline, cols))

        else:
            transformers.append((f'no_reconocida_{t}_{len(transformers)}', 'passthrough', cols)) 

    # Pipeline para datos categóricos ordinales
    if cat_ord_cols:
        ord_cols = list(cat_ord_cols.keys())
        ord_categories = [cat_ord_cols[col]['categories'] for col in ord_cols]
        ord_pipeline = Pipeline([('ord_encoder', OrdinalEncoder(categories=ord_categories))])
        transformers.append((f'ord_encoder_{len(transformers)}', ord_pipeline, ord_cols))

    # Pipeline para datos categóricos nominales ohe_dop
    if cat_nom_ohe_drop:
        ohe_drop_config = preprocessor_config.get('onehot_drop', {})
        cat_ohe_drop_pipeline = Pipeline([('ohe_drop', OneHotEncoder(
            drop = ohe_drop_config.get('drop', 'first'),
            handle_unknown = ohe_drop_config.get('handle_unknown', 'ignore'),
            sparse_output = False))
            ])
        transformers.append((f'ohe_drop_{len(transformers)}', cat_ohe_drop_pipeline, cat_nom_ohe_drop))
    
    # Pipeline para datos categoricos nominales ohe
    if cat_nom_ohe:
        ohe_config = preprocessor_config.get('onehot', {})
        ohe_pipeline = Pipeline([('ohe', OneHotEncoder(
            handle_unknown = ohe_config.get('handle_unknown', 'ignore'),
            sparse_output = False))
            ])
        transformers.append((f'ohe_{len(transformers)}', ohe_pipeline, cat_nom_ohe))


    processor = ColumnTransformer(
        transformers = transformers,
        remainder = preprocessor_config.get('remainder', 'drop')
    )
    return processor

def load_class(class_path):
    '''
    Se importa el modelo qeu se va a usar para base en los entrenamientos del BaggigClassifier
    '''
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    print(module)
    print(class_name)
    return getattr(module, class_name)

def build_model(model_config, seed_config):
    '''
    COnstruye el modelo de bagging con el archivo de configuración, además se importa el modelo base 
    Args:
        models_config (dict): Configuración para los modelos utilizados
        seed_config (int): Semilla para trazabilidad de los experimentos
    '''
    base_model_params_config = model_config.get('base_model_params', {})
    class_path = base_model_params_config.get('class', None)
    params_path = base_model_params_config.get('params', {})
    seed = seed_config

    base_class = load_class(class_path)
    base_estimator = base_class(**params_path)

    bagging = BaggingClassifier(
        estimator = base_estimator,
        n_estimators = model_config.get('n_estimators', 100),
        max_samples = model_config.get('max_samples', 0.3),
        bootstrap = model_config.get('bootstrap', True),
        random_state = seed,
        n_jobs = model_config.get('n_jobs', -1)
    )
    return bagging

def build_full_pipeline(config, seed):
    """
    COntruye el pipeline completo usando las funciones de preprocesamiento y la del modelo
    Args:
        config (dict): Es el archivo de configuración
        seed (int): Semilla para la trazabilidad
    """
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

    if feature_engineering_config.get('create_has_delinquency_history', False):
        columnas_config['numeric_cols'] = columnas_config.get('numeric_cols', []) + ['has_delinquency_story']
    
    if feature_engineering_config.get('create_severity_score', False):
        columnas_config['numeric_cols'] = columnas_config.get('numeric_cols', []) + ['severity_score']

    if feature_engineering_config.get('create_payment_income', False):
        columnas_config['numeric_cols'] = columnas_config.get('numeric_cols', []) + ['payment_income']

    # componentes del pipeline
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