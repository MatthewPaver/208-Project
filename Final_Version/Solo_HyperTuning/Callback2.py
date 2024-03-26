from keras import callbacks
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np


class MyCallback2(callbacks.Callback):
    def __init__(self, save_directory):
        super().__init__()
        self.save_directory = save_directory

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
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
