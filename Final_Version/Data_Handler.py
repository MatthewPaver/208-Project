"""
Module handles loading all datasets

This module currently loads the RockPaperScissors dataset but will be adapted
in the future in order to load and preprocess our custom dataset
"""

import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import numpy as np

IMAGE_DIMENSIONS = (128, 128)
PATH = str(Path(__file__).parent.parent) + "/data_pre_processing/processed_images/train"


def load_dataset():
    """
    This loads the training area of the dataset, setting batch size equal to the hyperparameter
    value as passed in the run_trial function in HyperCGAN if the path to data_pre_processing
    changes so that the relative path below data_pre_processing changes, the relative path
    will need to be adjusted
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
    I have also taken the liberty of adding a load_test_dataset function, for when we want to
    evaluate the models performance
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
