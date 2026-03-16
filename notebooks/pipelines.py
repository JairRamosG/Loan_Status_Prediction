def build_preprocessor(columnas_config, preprocessor_config):
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