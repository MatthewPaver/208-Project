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
def image_preprocessor1(image_shape = (128,128,3)):
    x = layers.Input(shape=image_shape)
    return x # when calling the model from the training loop, images and labels are passed automatically and do not need to be referenced in function calls.

def label_preprocessor1(): # in order for keras concatenation layer to work, both inputs must have the same dimensionality
    x = layers.Input(shape=(1,)) # this expresses the expected shape of this input stream, if it differs the model will not work (or you can change the shape)
    x = layers.Embedding(3,50)(x) # an embedding layer converts the declarative input into a tensor (so it can be reshaped) 50 refers to the size of the output, 3 is the input
    # dimension, and I frankly don't know what that is (but three seems the standard value).
    x = layers.Dense((128*128*3)) (x) # 128,128,3 is the shape of the image input using the tensorflow rock-paper-scissors dataset
    return x # in order for keras concatenation layer to work, both inputs must have the same dimensionality

def create_discriminator(): # in keras forward propagation is handled automatically when one calls 'discriminator' with input data
    stream1_input = image_preprocessor() #keras simply assigns the first dimension of the input tensor (when the model is called) to the first input layer it encounters, and so on.
    stream2_input = label_preprocessor() # it is therefore very important that input is passed image,label and not label,image.
    x = layers.Concatenate() ([stream1_input,stream2_input])

    x = layers.Conv2D(64,4) (x)
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

    model = tf.keras.Model([stream1_input, stream2_input], x)

    return model
