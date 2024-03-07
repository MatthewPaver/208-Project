from keras_tuner import HyperModel
from tensorflow.keras import Model
from Models.CGAN import CGAN
from Models.Generator import build_generator
from Models.Discriminator import build_discriminator
from tensorflow.keras.optimizers import Adam

class HyperCGAN(HyperModel):
    def build(self, hp) -> Model:
        latent_dim = hp.Choice('Latent Dim', [100])
        generator = build_generator(latent_dim)
        discriminator = build_discriminator()

        cgan = CGAN(
            generator=generator,
            discriminator=discriminator,
        )

        lr1 = hp.Choice('Generator LR', [1e-3])
        lr2 = hp.Choice('Discriminator LR', [1e-3])

        gen_optim = Adam(learning_rate = lr1, beta_1 = 0.5, beta_2 = 0.999 )
        disc_optim = Adam(learning_rate = lr2, beta_1 = 0.5, beta_2 = 0.999 )

        cgan.compile(
            gen_optimiser= gen_optim,
            disc_optimiser=disc_optim,
        )
        return cgan

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice('Batch Size', [128])
        return model.fit(*args, **kwargs, batch_size=batch_size)