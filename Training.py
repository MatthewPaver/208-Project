from Models import Discriminator
from Models import Generator
import tensorflow as tf
from tensorflow.data import Dataset
import DataHandler
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds
from matplotlib import gridspec


#Method parameters not stubbed as dependent on implementation
#image input may need normalising to (128,128,3) as part of the dataloader
def train_one_epoch(dataset: Dataset) -> None:
    for batch,labels in dataset:
        random_image_noise = np.random.rand(128,128,128,3) #I am assuming a batch size of 128
        random_image_noise = tf.convert_to_tensor(random_image_noise, dtype=tf.float32)
        fake_images = generator([random_image_noise,labels])
        generator.compile('Adam','binary_crossentropy')
        discriminator.compile('Adam','binary_crossentropy')
        discriminator.fit((batch,labels),np.ones(128,),128,1)
        discriminator.fit((fake_images,labels),np.zeros(128,),128,1)
        generator.fit((random_image_noise,target),np.ones((128,3)),128,1) # this assumes that the generator generates three images in response to each vector, if it only generates 1
        # just change the shape of np.one to (128,1)
    #TODO: Implement training for both nets for one epoch
    #TODO: Parameterize batch_size, image height and width as they are unknown parameters right now
    return

def train(dataset: Dataset, epochs=5) -> None:
    for epoch in range(1,epochs):
        train_one_epoch(dataset)
    #TODO: Implement training loops to call train_one_epoch and do any needed setup
    #TODO: Parameterize epochs as they are a hyper parameter
    return


if __name__ == "__main__":
    discriminator = Discriminator.create_discriminator()
    generator = Generator.create_generator()
    dataset = DataHandler.load_dataset()
    train(dataset,5)
    #TODO: Call methods to setup and begin training. This is equivalent of the main method in java
    #If statement used to dictate only main thread can execute not worker threads
