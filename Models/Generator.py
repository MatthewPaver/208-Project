from tensorflow.keras import layers
import tensorflow as tf

def normalized_tanh(x):
    """
    Defines Tanh function normalised for positive values only

    :param x: Layer to be passed into normalised tanh
    :return: Result after passing through the layer
    """
    return (tf.tanh(x) + 1) / 2

def build_generator(latent_dim=100):
    """
    Defines the architecture for the generator

    :param latent_dim: The size of the latent vector that will be inputted
    :return: The fully constructed model
    """
    #Activation function will be Tanh and ELU (Can change to Leaky ReLU/ReLU after further experiments)
    #Strides are doubling the input size
    #Batch normalization to ensure smooth training
    #Tanh is typically better and outputs to [-1, 1] but using a normalized tanh function it outputs [0,1] instead

    inputs1 = layers.Input(shape=(latent_dim,))
    x = layers.Dense(512*4*4)(inputs1)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    input_stream1 = layers.Reshape((4,4,512))(x) #Upsamples latent vector (B, 4, 4, 512) B = batch_size

    inputs2 = layers.Input(shape=(1,))
    x = layers.Embedding(3,50)(inputs2) #Encoding the label as a tensor
    x = layers.Dense((4*4)) (x)
    input_stream2 = layers.Reshape((4,4,1))(x) #Reshapes tensor to be the same shape as the image
    x = layers.Concatenate() ([input_stream1,input_stream2])  #Adds tensor as an extra colour channel in the image
    # (B, 4, 4, 513)
    
    x = layers.Conv2DTranspose(64*16, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # (B, 8, 8, 256)
   
    
    x = layers.Conv2DTranspose(64*8, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # (B, 16, 16, 128)

    x = layers.Conv2DTranspose(64*4, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # (B, 32, 32, 64)

    x = layers.Conv2DTranspose(64*2, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # (B, 64, 64, 32)

    x = layers.Conv2DTranspose(64*1, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # (B, 128, 128, 16)
    
    
    outputs = layers.Conv2D(3, kernel_size=4, padding='same', activation=normalized_tanh)(x)
    # (B, 128, 128, 3)
    
    
    model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=outputs, name='generator')
    return model
