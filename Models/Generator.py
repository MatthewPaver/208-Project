from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def normalized_tanh(x):
    return (tf.tanh(x) + 1) / 2

def image_preprocessing(latent_dim1 = 100):
    inputs = layers.Input(shape=(latent_dim1,))
    x = layers.Dense(512*4*4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Reshape((4,4,512))(x)
    return x


def tag_preprocessing():
    x = layers.Input(shape=(1,))
    x = layers.Embedding(3,50)(x)
    x = layers.Dense((4*4)) (x)
    x = layers.Reshape((4,4,1))(x)
    return x 


def build_generator(latent_dim): 
    input_stream2 = tag_preprocessing()
    input_stream1 = image_preprocessing()
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

# Define latent dimension
#latent_dim = 128

# Build the generator
generator = build_generator(latent_dim)

# Summary of the generator model
generator.summary()

