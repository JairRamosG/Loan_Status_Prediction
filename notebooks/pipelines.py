from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from utils import FeatureEngineering
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
    cat_nom_frec = columnas_config.get('cat_nom_frec', [])

    transformers = []

    # Pipeline para datos numéricos
    transform_groups = {}
    for col, conf in num_cols.items():
        t = conf.get('transform', 'passthrough')
        transform_groups.setdefault(t, []).append(col)
    
    for t, cols in transform_groups.items():
        if t == 'passthrough':
            transformers.append((f'num_passthrough', 'passthrough', cols))
        elif t == 'scale':
            pipeline = Pipeline([('scaler', StandardScaler())])
            transformers.append((f'scale', pipeline, cols))
        elif t == 'log_scale':
            num_pipeline = Pipeline([('log', FunctionTransformer(np.log, validate=True)),
                                    ('scaler', StandardScaler())])
            transformers.append((f'log_scale', num_pipeline, cols))
        elif t == 'log1p_scale':
            pipeline = Pipeline([('log1p_scale', FunctionTransformer(np.log1p, validate= True))])
            transformers.append(('log1p_scale', num_pipeline, cols))
        else:
            transformers.append(('no_reconocida', 'passthrough', cols)) 

    # Pipeline para datos categóricos ordinales
    if cat_ord_cols:
        ord_cols = list(cat_ord_cols.keys())
        ord_categories = [cat_ord_cols[col]['categories'] for col in ord_cols]
        ord_pipeline = Pipeline([('ord_encoder', OrdinalEncoder(categories=ord_categories))])
        transformers.append(('ord_encoder', ord_pipeline, cat_ord_cols))

    # Pipeline para datos categóricos nominales ohe
    if cat_nom_ohe_drop:
        ohe_drop_config = preprocessor_config.get('onehot_drop', {})
        cat_ohe_drop_pipeline = Pipeline([('ohe_drop', OneHotEncoder(
            drop = ohe_drop_config.get('drop', 'first'),
            handle_unknown = ohe_drop_config.get('handle_unknown', 'ignore'),
            sparse_output = False))
            ])
        transformers.append(('ohe_drop', cat_ohe_drop_pipeline, cat_nom_ohe_drop))

    processor = ColumnTransformer(
        transformers = transformers,
        remainder = preprocessor_config.get('remainder', 'drop')
    )
    return processor

def build_model(models_config, seed):
    pass

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

    if feature_engineering_config.get('create_has_delinquency_story', False):
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