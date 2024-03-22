"""
Module handles loading all datasets

This module currently loads the RockPaperScissors dataset but will be adapted
in the future in order to load and preprocess our custom dataset
"""

import tensorflow_datasets as tfds
import tensorflow as tf

IMAGE_DIMENSIONS = (128,128)

#TODO: Load and process custom dataset

def load_dataset():
    """
    Loads the RockPaperScissors Dataset and normalises it between 0 and 1.

    :return: A tuple containing images and labels. Order -> images, labels
    """
    ds = tfds.load('RockPaperScissors', split='train', as_supervised=True, shuffle_files=True)
    images = []
    labels = []
    for x,y in ds:
        x = tf.image.resize(x, IMAGE_DIMENSIONS)
        x = x / 255
        images.append(x)
        labels.append(y)
    images = images[:20]
    labels = labels[:20]
    images = tf.stack(images)
    labels = tf.stack(labels)
    return images, labels
