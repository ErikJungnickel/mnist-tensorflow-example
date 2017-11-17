import numpy as np
import keras.models
from keras.models import load_model
from scipy.misc import imread, imresize, imshow
import tensorflow as tf


def init():
    model = load_model('./model/model.h5')

    graph = tf.get_default_graph()

    return model, graph
