"""
The hypermodel class a CGAN

This module contains the hypermodel class for building a CGAN based on a HyperParameter
object passed to it. It contains a build method to construct it and a fit method to edit
the training process before each epoch
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras_tuner import HyperModel
from Models import Generator, Discriminator, CGAN


class HyperCGAN(HyperModel):
    def build(self, hp) -> Model:
        """
        Builds an instance of the CGAN model

        This overwrites the build method from keras_tuner.HyperModel. It is used to vary
        parameters during construction of a CGAN based on what hyperparameter values are chosen
        for the current trial

        :param hp: A keras_tuner HyperParameter object that defines all values to use in this trial
        :return: Returns a CGAN model
        """
        latent_dim = hp['Latent Dim']
        generator = Generator.build_generator(latent_dim)
        discriminator = Discriminator.build_discriminator()

        cgan = CGAN.CGAN(
            generator=generator,
            discriminator=discriminator,
        )
        lr1 = hp['Generator LR']
        lr2 = hp['Discriminator LR']

        gen_optim = Adam(learning_rate=lr1, beta_1=0.5, beta_2=0.999)
        disc_optim = Adam(learning_rate=lr2, beta_1=0.5, beta_2=0.999)

        cgan.compile(
            gen_optimiser=gen_optim,
            disc_optimiser=disc_optim,
        )
        return cgan

    def fit(self, hp, model, *args, **kwargs):
        """
        Trains the model but allows definable batch_size

        Overwrites method from keras_tuner.HyperModel. Trains the model on an epoch of the data
        but this method allows the tuner to include batch size as a tunable hyperparameter

        :param hp: A keras_tuner HyperParameter object that defines all values to use in this
        trial, including batch_size
        :param model: An instance of the CGAN model class
        :param args: Refer to superclass for extra information
        :param kwargs: Refer to superclass for extra information
        :return: Refer to superclass for extra information
        """
        batch_size = hp['Batch Size']
        return model.fit(*args, **kwargs, batch_size=batch_size)
