"""
Contains the CGAN model class

This module defines how a CGAN will function with the 2 different models composing it
as well as a custom training process along with custom loss functions and metric
reporting. generator_loss and discriminator_loss are functions but shouldn't be
used outside this file while the CGAN class itself contains init, compile and a
dynamic property called metrics that can be used by other code
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
from keras import metrics


def generator_loss(fake_output):
    """
    Calculates the total generator loss over the batch

    :param fake_output: The discriminator's output when fed the generated images conditioned on
    the labels
    :return: The calculated loss
    """
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    """
    Calculates the total discriminators loss over the batch

    :param real_output: The discriminator's output when fed the real images conditioned on the
    labels
    :param fake_output: The discriminator's output when fed the generated images conditioned on
    the labels
    :return: The calculated loss
    """
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


class CGAN(Model):
    def __init__(self, generator: Model , discriminator: Model) -> None:
        """
        Instantiates an instance of a CGAN

        :param generator: An instance of the Generator model class
        :param discriminator: An instance of the Generator model class
        """

        super(CGAN,self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.d_optimiser = None
        self.g_optimiser = None
        self.d_loss = metrics.Mean(name="Discriminator Loss")
        self.g_loss = metrics.Mean(name="Generator Loss")


    def compile(self, gen_optimiser: Optimizer, disc_optimiser: Optimizer) -> None:
        """
        Adds optimisers for both models

        Overwrites the compile method from tf.keras.Model superclass. Saves optimisers
        as attributes to use later. Need to call this method before training or tuning

        :param gen_optimiser: A keras optimiser to use for the generator
        :param disc_optimiser: A keras optimiser to use for the discriminator
        """

        super(CGAN, self).compile()
        self.g_optimiser = gen_optimiser
        self.d_optimiser = disc_optimiser


    def train_step(self, data):
        """
        Defines how training of one batch and backpropagation works

        Overwrites train_step from tf.keras.Model. Gets the output from the model in a forward
        pass, calculates the loss for both the generator and discriminator and then performs a
        backwards pass updating the gradient values

        :param data: A batch of images and labels [images,labels]
        :return: A tuple of the average losses over the batch for the generator and discriminator.
         Order -> generator loss, discriminator loss
        """

        images , labels = data
        batch_size = tf.shape(images)[0]
        latent_dim = 100

        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator([noise, labels],training=True)

            real_output = self.discriminator([images,labels], training= True)
            fake_output = self.discriminator([generated_images, labels], training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimiser.apply_gradients(zip(gradients_of_gen, self.generator.trainable_variables))
        self.d_optimiser.apply_gradients(zip(gradients_of_disc, self.discriminator.trainable_variables))

        batch_size = tf.cast(batch_size, tf.float32)
        g_loss = gen_loss/ batch_size
        d_loss = disc_loss/ batch_size

        self.d_loss.update_state(d_loss)
        self.g_loss.update_state(g_loss)

        return {"Generator Loss": self.g_loss.result(), "Discriminator Loss": self.d_loss.result()}

    @property
    def metrics(self):
        """
        A property dynamically set to the loss values. Needed to prevent error at end of epoch
        
        :return: Tuple of discriminator loss and generator loss
        """""
        return [self.d_loss, self.g_loss]
