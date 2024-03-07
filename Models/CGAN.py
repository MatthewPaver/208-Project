from keras.models import Model
from tensorflow.keras.optimizers import Optimizer


#TODO: Define training step, calculate losses, track metrics
class CGAN(Model):
    def __init__(self, generator: Model , discriminator: Model) -> None:
        super(CGAN,self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.d_optimiser = None
        self.g_optimiser = None

    def train_step(self, data):
        return

    def compile(self, gen_optimiser: Optimizer, disc_optimiser: Optimizer) -> None:
        self.g_optimiser = gen_optimiser
        self.d_optimiser = disc_optimiser