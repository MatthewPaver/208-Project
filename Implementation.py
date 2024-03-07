from DataHandler import load_dataset
from Models.Generator import build_generator
from Models.Discriminator import build_discriminator
from Models.CGAN import CGAN
from keras_tuner import tuners
from Models.HyperCGAN import HyperCGAN


#TODO: Change objective to be a tracked metric

if __name__ == "__main__":
    dataset = load_dataset()
    gen = build_generator(100)
    disc = build_discriminator()
    cgan = CGAN(generator=gen, discriminator=disc)
    tuner = tuners.GridSearch(
        hypermodel=HyperCGAN(),
        objective="FILL IN OBJECTIVE",
    )


    tuner.search(dataset, epochs=5)

