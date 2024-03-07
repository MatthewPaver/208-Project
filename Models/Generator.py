from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def normalized_tanh(x):
    return (tf.tanh(x) + 1) / 2




# however changed this code back to an earlier version last night, be aware that I had to re-debug it.
# inputs1 and inputs2 have to be outside the function calls so that they can be refered to as the inputs for the model in its declaration
# otherwise tensroflow expects the shape of the inputs to be 4,4,512, (which is the shape of the input after pre-processing,
# and not 128,128,3 which is their shape when they are passed to the funcion.
# if these layers are inside the pre-processing functions, the code will crash.
def build_generator(latent_dim):
    #pre-processing for first input stream
    inputs1 = layers.Input(shape=(latent_dim,))
    x = layers.Dense(512*4*4)(inputs1)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    input_stream1 = layers.Reshape((4,4,512))(x)
    #pre_processing for second input stream
    inputs2 = layers.Input(shape=(1,))
    x = layers.Embedding(3,50)(inputs2)
    x = layers.Dense((4*4)) (x)
    input_stream2 = layers.Reshape((4,4,1))(x)
    #input_stream1 = image_preprocessing()
    x = layers.Concatenate() ([input_stream1,input_stream2])
    
    #Activation function will be Tanh and ELU (Can change to Leaky ReLU/ReLU after further experiments)
    #Strides are doubling the input size
    #Batch normalization to ensure smooth training
    #Tanh is typically better and outputs to [-1, 1] but using a normlized tanh function it outputs [0,1] instead
    
    x = layers.Conv2DTranspose(64*16, kernel_size=4, strides=2, padding='same')(x)
    #8 x 8
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
   
    
    x = layers.Conv2DTranspose(64*8, kernel_size=4, strides=2, padding='same')(x)
    #16 x 16
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Conv2DTranspose(64*4, kernel_size=4, strides=2, padding='same')(x)
    #32 x 32
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Conv2DTranspose(64*2, kernel_size=4, strides=2, padding='same')(x)
    #64 x 64
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Conv2DTranspose(64*1, kernel_size=4, strides=2, padding='same')(x)
    #128 x 128
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    
    
    outputs = layers.Conv2D(3, kernel_size=4, padding='same', activation=normalized_tanh)(x)
    
    
    model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=outputs, name='generator')
    return model
