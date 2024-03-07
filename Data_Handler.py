from tensorflow.keras import datasets

IMAGE_DIMENSIONS = 128

def load_dataset():
    """
    Loads the MNIST Dataset and normalises it between 0 and 1

    :return: A tuple containing images and labels. Order -> images, labels
    """
    train, _ = datasets.fashion_mnist.load_data()
    x, y = train
    x = x/255
    return x, y
