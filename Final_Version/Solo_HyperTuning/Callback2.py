"""
This module contains a custom callback that's used to save the models output for
all 9 classes of the custom dataset
"""

from keras import callbacks
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np


class MyCallback2(callbacks.Callback):
    def __init__(self, save_directory):
        """
        Instantiates the callback with an additional save_directory attribute

        :param save_directory: The path to save the weights to
        """
        super().__init__()
        self.save_directory = save_directory

    def on_epoch_end(self, epoch, logs=None):
        """
        Finds the last saved epoch by calling find_latest_epoch. Using that it saves an image
        which is a grid of the 9 separate outputs of the model based on the 9 custom classes
        of the dataset. This is saved in the trials directory with the name
        epoch_{epoch}_grid.png
        """
        epoch = self.find_latest_epoch()
        latent_dim = 100
        num_classes = 9
        generated_images = []

        fig, axs = plt.subplots(3, 3, figsize=(9, 9))

        for i in range(num_classes):
            noise = tf.random.normal([1, latent_dim])
            class_label = tf.expand_dims(tf.constant(i), axis=-1)
            generated_image = self.model.generator([noise, class_label], training=False)
            generated_images.append(generated_image)

        for i, generated_image in enumerate(generated_images):
            generated_image = np.squeeze(generated_image, axis=0)
            ax = axs[i // 3, i % 3]
            ax.imshow(generated_image)
            ax.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        path = os.path.join(self.save_directory, f"epoch_{epoch}_grid.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def find_latest_epoch(self):
        """
        Searches the trials directory for the last fully saved epoch

        :return: The last saved epoch as an Int
        """
        epochs_seen = []
        if os.path.exists(self.save_directory):
            for file in os.listdir(self.save_directory):
                if file.startswith("generator_epoch_"):
                    epochs_seen.append(int(file.split("_")[2].split(".")[0]))
            if epochs_seen:
                return max(epochs_seen)
        return 0
