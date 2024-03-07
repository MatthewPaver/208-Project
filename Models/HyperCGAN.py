from keras_tuner import HyperModel
from tensorflow.keras import Model
from Models.CGAN import CGAN
from Models.Generator import build_generator
from Models.Discriminator import build_discriminator

#TODO: Create and call functions to define the optimisers
class HyperCGAN(HyperModel):
    def build(self, hp) -> Model:
        latent_dim = hp.Choice('Latent Dim', [100])
        generator = build_generator(latent_dim)
        discriminator = build_discriminator()

        cgan = CGAN(
            generator=generator,
            discriminator=discriminator,
        )
        cgan.compile(
            gen_optimiser=,
            disc_optimiser=,
        )
        return cgan

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice('Batch Size', [128])
        return model.fit(*args, **kwargs, batch_size=batch_size)