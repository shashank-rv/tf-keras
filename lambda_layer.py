#working with lambda layer in keras
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys


#model: (CNN):

layer_1= tf.keras.layers.Input(shape = (1024,1024,3), name="layer1")

def custom_layer(tensor):
    return tensor + 2    # All the tensor values in layer_1 will be incremented by 2.

lambda_layer = tf.keras.layers.Lambda(custom_layer, name="lambda_layer")(layer_1)


model = tf.keras.models.Model(layer_1, lambda_layer, name="model")

model.summary()