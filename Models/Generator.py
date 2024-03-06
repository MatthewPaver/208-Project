from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def normalized_tanh(x):
    return (tf.tanh(x) + 1) / 2

def image_preprocessing(image_shape = (128,128,3)) -> tf.Tensor:


def build_generator(latent_dim): 
    inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(8*8*128)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Reshape((8, 8, 128))(x)
    
    #Activation function will be Tanh and ELU (Can change to Leaky ReLU/ReLU after further experiments)
    #Strides are doubling the input size
    #Batch normalization to ensure smooth training
    #Tanh is typically better and outputs to [-1, 1] but using a normlized tanh function it outputs [0,1] instead
    
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    #16 x 16
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
   
    
    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
    #32 x 32
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same')(x)
    #64 x 64
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    
    
   
    
    
    outputs = layers.Conv2D(3, kernel_size=4, padding='same', activation=normalized_tanh)(x)

    model = Model(inputs=inputs, outputs=outputs, name='generator')
    return model

# Define latent dimension
latent_dim = 128

# Build the generator
generator = build_generator(latent_dim)

# Summary of the generator model
generator.summary()

