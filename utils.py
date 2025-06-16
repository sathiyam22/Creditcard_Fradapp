def save_model(model, model_name):
    model.save(model_name)

def load_model(model_name):
    from tensorflow.keras.models import load_model
    return load_model(model_name)

def log_message(message):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(message)

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)