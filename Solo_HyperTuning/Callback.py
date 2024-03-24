"""
This module contains a custom callback that's used to save model weights each epoch
"""

from keras import callbacks
import os
import pickle


class MyCallback(callbacks.Callback):
    def __init__(self, save_directory):
        """
        Instantiates the callback with an addition save_directory attribute

        :param save_directory: The path to save the weights to
        """
        super().__init__()
        self.save_directory = save_directory

    def on_epoch_end(self, epoch, logs=None):
        """
        Saves the current states of all parts of CGAN

        This method saves the model weights of the generator and discriminator in the .h5
        format and the current state of the two optimisers as pickle dump files (.pkl)
        all files follow patter {part_name}_{epoch}.{file extension}

        :param epoch: Int value of the current epoch
        :param logs: Not used, added only to match superclass method signature
        """
        epoch = self.find_latest_epoch()
        epoch = epoch + 1
        print(f"save dir: {self.save_directory} + {epoch}")
        generator_filename = os.path.join(self.save_directory, f"generator_epoch_{epoch}.h5")
        discriminator_filename = os.path.join(self.save_directory, f"discriminator_epoch_{epoch}.h5")
        generator_optimiser_filename = os.path.join(self.save_directory, f"generator_optimiser_epoch_{epoch}.pkl")
        discriminator_optimiser_filename = os.path.join(self.save_directory, f"discriminator_optimiser_epoch_{epoch}.pkl")

        os.makedirs(self.save_directory, exist_ok=True)

        self.model.generator.save_weights(generator_filename)
        self.model.discriminator.save_weights(discriminator_filename)

        with open(generator_optimiser_filename, 'wb') as f:
            pickle.dump(self.model.g_optimiser.get_config(), f)

        with open(discriminator_optimiser_filename, 'wb') as f:
            pickle.dump(self.model.d_optimiser.get_config(), f)

        print(f"\n Models and optimizers saved for epoch {epoch}.")

    def find_latest_epoch(self):
        epochs_seen = []
        if os.path.exists(self.save_directory):
            for file in os.listdir(self.save_directory):
                if file.startswith("generator_epoch_"):
                    epochs_seen.append(int(file.split("_")[2].split(".")[0]))
            if epochs_seen:
                return max(epochs_seen)
        return 0
