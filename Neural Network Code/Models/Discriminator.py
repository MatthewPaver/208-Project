"""
This module handles building of the discriminator model
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_discriminator():
    """
    Defines and builds the architecture for the discriminator

    :return: The built model
    """
    img_shape = (64, 64, 3)
    con_label = layers.Input(shape=(1,))
    # B = batch_size
    # (B, 1)

    label_embedding = layers.Embedding(5, 10)(con_label)
    label_embedding = layers.Flatten()(label_embedding)
    # (B, 10)

    label_embedding = layers.Dense(img_shape[0] * img_shape[1])(label_embedding)
    label_embedding = layers.Reshape((img_shape[0], img_shape[1], 1))(label_embedding)
    # (B, 64, 64, 1)

    stream1_input = layers.Input(shape=(64, 64, 3))
    merge = layers.Concatenate()([stream1_input, label_embedding])
    # (B, 64, 64, 4)

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merge)
    x = layers.LeakyReLU()(x)
    # (B, 32, 32, 64)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    # (B, 16, 16, 128)

    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    # (B, 8, 8, 256)

    x = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    # (B, 4, 4, 512)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    # (B, 8192)

    x = layers.Dense(1, dtype='float32')(x)
    # (B, 1)

    model = tf.keras.Model([stream1_input, con_label], x)
    return model
