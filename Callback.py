from keras import callbacks
import os
import pickle

class MyCallback(callbacks.Callback):
    def __init__(self, save_directory):
        super().__init__()
        self.save_directory = save_directory

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        generator_filename = os.path.join(self.save_directory, f"generator_epoch_{epoch}.h5")
        discriminator_filename = os.path.join(self.save_directory, f"discriminator_epoch_{epoch}.h5")
        generator_optimizer_filename = os.path.join(self.save_directory, f"/generator_optimizer_epoch_{epoch}.pkl")
        discriminator_optimizer_filename = os.path.join(self.save_directory, f"discriminator_optimizer_epoch_{epoch}.pkl")

        os.makedirs(self.save_directory, exist_ok=True)

        self.model.generator.save_weights(generator_filename)
        self.model.discriminator.save_weights(discriminator_filename)

        with open(generator_optimizer_filename, 'wb') as f:
            pickle.dump(self.model.generator_optimizer.get_config(), f)

        with open(discriminator_optimizer_filename, 'wb') as f:
            pickle.dump(self.model.discriminator_optimizer.get_config(), f)

        print(f"Models and optimizers saved for epoch {epoch}.")