from tensorflow.keras import datasets
import tensorflow_datasets as tfds
import tensorflow as tf

IMAGE_DIMENSIONS = (128,128)

def load_dataset():
    """
    Loads the MNIST Dataset and normalises it between 0 and 1

    :return: A tuple containing images and labels. Order -> images, labels
    """
    ds = tfds.load('RockPaperScissors', split='train', as_supervised=True, shuffle_files=True)
    images = []
    labels = []
    for x,y in ds:
        x = tf.image.resize(x, IMAGE_DIMENSIONS)
        x = x/255
        images.append(x)
        labels.append(y)
    images = tf.stack(images)
    labels = tf.stack(labels)
    return images, labels
