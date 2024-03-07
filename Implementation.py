import DataHandler
from Models import Generator
from Models import Discriminator

dataset = DataHandler.load_dataset()
gen = Generator.build_generator(100)
disc = Discriminator.build_discriminator()

