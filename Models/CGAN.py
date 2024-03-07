from keras.models import Model
from tensorflow.keras.optimizers import Optimizer


#TODO: Define training step, calculate losses, track metrics
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

    def compile(self, gen_optimiser: Optimizer, disc_optimiser: Optimizer) -> None:
        """
        Adds optimisers for both models

        Overwrites the compile method from tf.keras.Model superclass. Saves optimisers as attributes to use later
        Need to call this method before training or tuning

        :param gen_optimiser: A keras optimiser to use for the generator
        :param disc_optimiser: A keras optimiser to use for the discriminator
        """

        self.g_optimiser = gen_optimiser
        self.d_optimiser = disc_optimiser

    def train_step(self, data):
        """
        Defines how training of one batch and backpropagation works

        :param data: The batch of data. [images, labels]
        :return:
        """
        return

