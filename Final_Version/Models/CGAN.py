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
import tensorflow as tf
from keras import metrics

logdir = "logs/scalars/6"
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()



class CGAN(Model):
    def __init__(self, generator: Model, discriminator: Model) -> None:
        """
        Instantiates an instance of a CGAN

        :param generator: An instance of the Generator model class
        :param discriminator: An instance of the Generator model class
        """

        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.d_optimiser = None
        self.g_optimiser = None
        self.d_loss = metrics.Mean(name="Discriminator Loss")
        self.g_loss = metrics.Mean(name="Generator Loss")
        self.batches = 0

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

    def gp(self, batch_size, real_images, fake_images, labels):
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, labels], training=True)
        grads = gp_tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, data):
        images, labels = data
        batch_size = tf.shape(images)[0]
        latent_dim = 100

        for _ in range(10):
            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator([noise, labels])

                real_output = self.discriminator([images, labels], training=True)
                fake_output = self.discriminator([generated_images, labels], training=True)

                real_loss = tf.reduce_mean(real_output)
                fake_loss = tf.reduce_mean(fake_output)
                gp = self.gp(batch_size, images, generated_images, labels)
                disc_loss = (fake_loss - real_loss) + (gp * 30)
                tf.summary.scalar('fake_loss', data=fake_loss, step=self.batches)
                tf.summary.scalar('real_loss', data=real_loss, step=self.batches)
                tf.summary.scalar('gp', data=gp, step=self.batches)

                self.batches += 1
            gradients_of_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.d_optimiser.apply_gradients(zip(gradients_of_disc, self.discriminator.trainable_variables))
            b_size = tf.cast(batch_size, dtype=tf.float32)
            disc_loss = disc_loss / b_size
            tf.summary.scalar('Disc loss', data=disc_loss, step=self.batches)

        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator([noise, labels], training=True)
            fo = self.discriminator([generated_images, labels], training=True)
            gen_loss = -tf.reduce_mean(fo)
        gradients_of_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimiser.apply_gradients(zip(gradients_of_gen, self.generator.trainable_variables))

        gen_loss = gen_loss / b_size
        self.d_loss.update_state(disc_loss)
        self.g_loss.update_state(gen_loss)

        tf.summary.scalar('Gen Loss', data=gen_loss, step=self.batches)
        return {"Generator Loss": self.g_loss.result(),
                "Discriminator Loss": self.d_loss.result()
                }

    @property
    def metrics(self):
        """
        A property dynamically set to the loss values. Needed to prevent error at end of epoch
        
        :return: Tuple of discriminator loss and generator loss
        """
        return [self.d_loss, self.g_loss]
