import tensorflow as tf
import keras.models

def init_session():
    cfg = tf.ConfigProto()

    cfg.gpu_options.allow_growth = True

    return tf.InteractiveSession()

def load_model(model_path):
    return keras.models.load_model(model_path, compile=False)
