"""
This module contains a custom callback that's used to save the models output for
all the classes of the custom dataset
"""

from keras import callbacks
import tensorflow as tf
import os
import numpy as np
import cv2


class MyCallback2(callbacks.Callback):
    def __init__(
            self,
            noise_dim: int,
            output_path: str,
            examples_to_generate: int = 5,
            grid_size: tuple = (5, 1),
            spacing: int = 5,
    ) -> None:
        super().__init__()
        self.seed = tf.random.normal([examples_to_generate, noise_dim])
        self.results_path = output_path + '/results'
        self.grid_size = grid_size
        self.spacing = spacing

        os.makedirs(self.results_path, exist_ok=True)

    def save_plt(self, epoch: int, results: np.ndarray):
        w, h, c = results[0].shape
        grid = np.zeros((self.grid_size[0] * w + (self.grid_size[0] - 1) * self.spacing,
                         self.grid_size[1] * h + (self.grid_size[1] - 1) * self.spacing, c), dtype=np.uint8)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                grid[i * (w + self.spacing):i * (w + self.spacing) + w,
                j * (h + self.spacing):j * (h + self.spacing) + h] = results[i * self.grid_size[1] + j]

        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'{self.results_path}/img_{epoch}.png', grid)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        labels = tf.constant([1, 2, 3, 4, 5])
        predictions = self.model.generator([self.seed, labels], training=False)
        predictions_uint8 = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)
        self.save_plt(epoch, predictions_uint8)

