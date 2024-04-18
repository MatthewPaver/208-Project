"""
This module handles building of the generator model
"""

from tensorflow.keras import layers
import tensorflow as tf


def build_generator(noise_dim):
    """
    Defines the model architecture. Strides and kernel size adapted to
    reduce artifacts

    :param noise_dim: The size of the latent vector to be passed to the model
    :return: The built model
    """
    inputs1 = layers.Input(shape=(noise_dim,))
    # B = batch_size
    # (B, noise_dim)

    label_input = layers.Input(shape=(1,))
    # (B, 1)

    label_embedding = layers.Embedding(5, 10)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    # (B, 10)

    concatenated_input = layers.Concatenate()([inputs1, label_embedding])
    x = layers.Dense(512 * 4 * 4, use_bias=False)(concatenated_input)
    x = layers.Reshape((4, 4, 512))(x)
    # (B, 4, 4, 512)

    x = layers.Conv2DTranspose(64 * 8, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    # (B, 8, 8, 512)

    x = layers.Conv2DTranspose(64 * 4, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    # (B, 16, 16, 256)

    x = layers.Conv2DTranspose(64 * 2, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    # (B, 32, 32, 128)

    x = layers.Conv2DTranspose(64 * 1, kernel_size=5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    # (B, 64, 64, 64)

    outputs = layers.Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
    # (B, 64, 64, 3)

    model = tf.keras.Model(inputs=[inputs1, label_input], outputs=outputs, name='generator')
    return model
