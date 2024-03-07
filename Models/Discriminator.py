import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras
from IPython import display
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds
from matplotlib import gridspec
import numpy

# please do not put <-ft.tensor after my declerations, as it defaults to requiring the tensor output of the pre-processors be the same size as the input functions, and create_discriminator
# does not return an ft.keras.Model object to the caller, it returns it to a tensorflow function, which in turn finds the calling object and overides the return, so requiring the function
# to directly return a model object crashes the code.

# I have changed the preprocessing functions in the discriminator to image_preprocessor1 and label_preprocessor1, so that they do nmot overload the definitions for the smamge functions
# in generator.py, when they are both imported to train.py
def label_preprocessor1(in_shape=(128,128,3)):
    con_label = layers.Input(shape=(1,))
    x = layers.Embedding(3, 50)(con_label)
    x = layers.Dense((128*128*3))(x)
    x = layers.Reshape((128, 128, 3))(x)
    return con_label, x


def image_preprocessor1(in_shape=(128,128,3)):
    inp_image = layers.Input(shape=in_shape)
    return inp_image

def create_discriminator():
    con_label, stream2_input = label_preprocessor1()
    stream1_input = image_preprocessor1()
    # concat label as a channel
    merge = layers.Concatenate()([stream1_input, stream2_input])
    
    x = layers.Conv2D(64,4) (merge)
    x = layers.BatchNormalization() (x)
    x = layers.ReLU() (x)

    x = layers.Conv2D(64,4) (x)
    x = layers.BatchNormalization() (x)
    x = layers.ReLU() (x)

    x = layers.Conv2D(64,4) (x)
    x = layers.BatchNormalization() (x)
    x = layers.ReLU() (x)

    x = layers.Flatten() (x)
    x = layers.Dropout(0.3) (x)
    x = layers.Dense(1,activation ='sigmoid') (x)

    model = tf.keras.Model([stream1_input, con_label], x)

    return model
