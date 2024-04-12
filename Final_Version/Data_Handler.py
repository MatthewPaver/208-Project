"""
Module handles loading our custom dataset for training and evaluation as well as a pre-made
dataset for testing purposes
"""

import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import numpy as np

IMAGE_DIMENSIONS = (128, 128)
PATH = str(Path(__file__).parent.parent) + "/data_pre_processing/processed_images/train"
PATH2 = str(Path(__file__).parent.parent) + "/data_pre_processing/test_augmented"


def load_dataset():
    """
    This method loads the training portion of our custom dataset. The images are resized based on
    the IMAGE_DIMENSIONS constant defined at the top of this file. The pixel values are then
    normalised to be between 0 and 1 to speed up the training process.

    :return: Images, Labels which are arrays where Labels[0] is the label for images[0]
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=PATH,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=None,
        image_size=IMAGE_DIMENSIONS,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    images = []
    labels = []

    for x, y in dataset:
        x = x / 255.0
        images.append(x)
        labels.append(y)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def load_test_dataset():
    """
    This method loads the test portion of our custom dataset. The images are resized based on
    the IMAGE_DIMENSIONS constant defined at the top of this file. The pixel values are then
    normalised to be between 0 and 1 to speed up the training process. This will be used
    to evaluate the trained models

    :return: Images, Labels which are arrays where Labels[0] is the label for images[0]
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=PATH2,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=None,
        image_size=IMAGE_DIMENSIONS,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    images = []
    labels = []

    for x, y in dataset:
        x = x / 255.0
        images.append(x)
        labels.append(y)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def load_dataset_rock_paper_scissors():
    """
    Loads the RockPaperScissors Dataset and normalises it between 0 and 1.

    :return: A tuple containing images and labels. Order -> images, labels
    """
    ds = tfds.load('RockPaperScissors', split='train', as_supervised=True, shuffle_files=True)
    images = []
    labels = []
    for x, y in ds:
        x = tf.image.resize(x, IMAGE_DIMENSIONS)
        x = x / 255
        images.append(x)
        labels.append(y)

    images = tf.stack(images)
    labels = tf.stack(labels)
    return images, labels
