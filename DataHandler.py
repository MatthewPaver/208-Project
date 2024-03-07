from tensorflow.data import Dataset
import tensorflow as tf

IMAGE_DIMENSIONS = 128

def load_dataset() -> Dataset:
    train, _ = tf.keras.datasets.fashion_mnist.load_data()
    x, y = train
    x = x/255
    return x, y