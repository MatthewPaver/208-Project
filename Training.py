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

learning_rate = 0.1

# this instantiates the optimizer for the generator, it is outside of the training loop because it only needs to run once, and takes a lot of time.
generator_optimizer = tf.keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.5, beta_2 = 0.999 )

#Method parameters not stubbed as dependent on implementation
#image input may need normalising to (128,128,3) as part of the dataloader
def train_one_epoch(dataset: Dataset) -> None:
    for batch,labels in dataset:
        random_image_noise = np.random.rand(128,128,128,3) #I am assuming a batch size of 128
        random_image_noise = tf.convert_to_tensor(random_image_noise, dtype=tf.float32)
        fake_images = generator([random_image_noise,labels])
        discriminator.compile('Adam','binary_crossentropy')
        tf.keras.set_value(discriminator.optimizer.learning_rate, learning_rate)
        discriminator.fit((batch,labels),np.ones(128,),128,1)
        discriminator.fit((fake_images,labels),np.zeros(128,),128,1)
        # the generator has a non standard loss function (using the loss of another model to update)
        # therefore we cannot use the .fit() function and must manually instantiate the optimizer and loss function
        with tf.GradientTape() as gen_tape:
            generated_images = conditional_gen([noise,target], training=True) # this is a forward pass of the generator before 
            fake_output = conditional_discriminator([generated_images,target], training=True) # this which is a forward pass of the discriminator
            real_targets = tf.ones_like(fake_output) # this generates an array of ones the size of the discriminator output so that
            gen_loss = binary_cross_entropy(real_targets, fake_output) # it can be compared with the values output from the discriminator to calculate loss
            #binary cross entropy is a pre-built loss function from the tensorflow core library
 
        gradients_of_gen = gen_tape.gradient(gen_loss, conditional_gen.trainable_variables) # the information recorded by the GradientTape() object is then applyed via a black box
        # tensorflow process to calculate updated weights for the generator
        generator_optimizer.apply_gradients(zip(gradients_of_gen, conditional_gen.trainable_variables)) # which is then backpropogated.   
    #TODO: Implement training for both nets for one epoch
    #TODO: Parameterize batch_size, image height and width as they are unknown parameters right now
    return

def train(dataset: Dataset, epochs=5) -> None:
    for epoch in range(1,epochs):
        train_one_epoch(dataset)
    #TODO: Implement training loops to call train_one_epoch and do any needed setup
    #TODO: Parameterize epochs as they are a hyper parameter
    return


if __name__ == "__main__":
    discriminator = Discriminator.create_discriminator()
    generator = Generator.create_generator()
    dataset = DataHandler.load_dataset()
    train(dataset,5)
    #TODO: Call methods to setup and begin training. This is equivalent of the main method in java
    #If statement used to dictate only main thread can execute not worker threads
