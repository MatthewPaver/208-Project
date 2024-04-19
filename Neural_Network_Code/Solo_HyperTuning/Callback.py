"""
This module contains a custom callback that's used to save model weights each epoch
"""

from keras import callbacks
import os
import pickle
import cv2
import tensorflow as tf
import numpy as np


class MyCallback(callbacks.Callback):
    def __init__(self, save_directory):
        super().__init__()
        self.save_directory = save_directory
        self.image_directory = save_directory + "/Images"
        os.makedirs(self.save_directory, exist_ok=True)
        os.makedirs(self.image_directory, exist_ok=True)
        self.seed = tf.random.normal([5, 128])
        self.grid_size = (5, 1)
        self.spacing = 5

    def on_epoch_end(self, _, logs=None):
        """
        Called at the end of each epoch. Calls other methods of this class

        :param _: Current epoch, placeholder used as it's not necessarily correct
        :param logs: A list of the results of the batches in the epoch
        """

        curr_epoch = self.find_latest_epoch
        self.save_the_weights(curr_epoch)
        self.save_the_images(curr_epoch)

    def save_the_weights(self, epoch):
        """
        Saves the weights in a way that allows pausing and resuming of training

        :param epoch: The current epoch to save weights for
        """

        print(f"save dir: {self.save_directory} + {epoch}")
        generator_filename = os.path.join(self.save_directory, f"generator_epoch_{epoch}.weights.h5")
        discriminator_filename = os.path.join(self.save_directory, f"discriminator_epoch_{epoch}.weights.h5")
        generator_optimiser_filename = os.path.join(self.save_directory, f"generator_optimiser_epoch_{epoch}.pkl")
        discriminator_optimiser_filename = os.path.join(self.save_directory, f"discriminator_optimiser_epoch_{epoch}.pkl")

        self.model.generator.save_weights(generator_filename)
        self.model.discriminator.save_weights(discriminator_filename)

        with open(generator_optimiser_filename, 'wb') as f:
            pickle.dump(self.model.g_optimiser.get_config(), f)

        with open(discriminator_optimiser_filename, 'wb') as f:
            pickle.dump(self.model.d_optimiser.get_config(), f)

        print(f"\n Models and optimizers saved for epoch {epoch}.")

    def save_the_images(self, epoch):
        """
        Saves an image for each class in one grid and saves the resulting image with the
        epoch number

        :param epoch: The current epoch
        """
        labels = tf.constant([1, 2, 3, 4, 5])
        generated_images = self.model.generator([self.seed, labels], training=False)
        images_as_RGB = (generated_images * 127.5 + 127.5).numpy().astype(np.uint8)
        w, h, c = images_as_RGB[0].shape
        grid = np.zeros((self.grid_size[0] * w + (self.grid_size[0] - 1) * self.spacing,
                         self.grid_size[1] * h + (self.grid_size[1] - 1) * self.spacing, c), dtype=np.uint8)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                grid[i * (w + self.spacing):i * (w + self.spacing) + w,
                j * (h + self.spacing):j * (h + self.spacing) + h] = images_as_RGB[i * self.grid_size[1] + j]

        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'{self.image_directory}/img_{epoch}.png', grid)

    @property
    def find_latest_epoch(self):
        """
        Searches for the latest epoch saved

        :return: The current epoch to be saved
        """
        epochs_seen = []
        if os.path.exists(self.save_directory):
            for file in os.listdir(self.save_directory):
                if file.startswith("generator_epoch_"):
                    epochs_seen.append(int(file.split("_")[2].split(".")[0]))
            if epochs_seen:
                return max(epochs_seen) + 1
        return 1
