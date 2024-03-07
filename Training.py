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
from tensorflow.keras import backend as K

#tuneable parameters
learning_rate1 = 0.1
epochs1 = 5
latent_dim = 100
batch_size = 128 # does nothing until dataloader active

# this instantiates the optimizer for the generator, it is outside the training loop because it only needs to run once, and takes a lot of time.
generator_optimizer = tf.keras.optimizers.Adam(lr = learning_rate1, beta_1 = 0.5, beta_2 = 0.999 )

#Method parameters not stubbed as dependent on implementation
#image input may need normalising to (128,128,3) as part of the dataloader
def train_one_epoch(dataset: Dataset) -> None:
    for batch,labels in dataset:
        random_image_noise = tf.random.normal([labels.shape[0], latent_dim])
        fake_images = generator([random_image_noise,labels])
        discriminator.compile('Adam','binary_crossentropy')
        K.set_value(discriminator.optimizer.learning_rate, learning_rate1)
        discriminator.fit((batch,labels),np.ones(128,),128,1)
        discriminator.fit((fake_images,labels),np.zeros(128,),128,1)
        # the generator has a non-standard loss function (using the loss of another model to update)
        # therefore we cannot use the .fit() function and must manually instantiate the optimizer and loss function
        with tf.GradientTape() as gen_tape:
            generated_images = generator([random_image_noise,labels], training=True) # this is a forward pass of the generator before 
            fake_output = discriminator([generated_images,labels], training=True) # this which is a forward pass of the discriminator
            real_targets = tf.ones_like(fake_output) # this generates an array of ones the size of the discriminator output so that
            gen_loss = binary_cross_entropy(real_targets, fake_output) # it can be compared with the values output from the discriminator to calculate loss
            #binary cross entropy is a pre-built loss function from the tensorflow core library
 
        gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables) # the information recorded by the GradientTape() object is then applied via a black box
        # tensorflow process to calculate updated weights for the generator
        generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables)) # which is then backpropogated.
    return

def train(dataset: Dataset, epochs=5) -> None:
    for epoch in range(1,epochs):
        train_one_epoch(dataset)
    return


if __name__ == "__main__":
    discriminator = Discriminator.build_discriminator()
    generator = Generator.build_generator(100)
    dataset = DataHandler.load_dataset()
    train(dataset,epochs1)
    #If statement used to dictate only main thread can execute not worker threads
