from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def build_generator(latent_dim): 
    inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(8*8*128)(inputs)
    x = layers.BatchNormalization()
    x = layers.Reshape((8, 8, 128))(x)
    
    
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()
    x = layers.ReLU()(x)
   
    
    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()
    x = layers.ReLU()(x)
   
    
    
    outputs = layers.Conv2D(3, kernel_size=4, padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='generator')
    return model

# Define latent dimension
latent_dim = 128

# Build the generator
generator = build_generator(latent_dim)

# Summary of the generator model
generator.summary()

#Tesing github push from vscode