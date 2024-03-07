import keras_tuner
from tensorflow.keras import Model
class HyperCGAN(keras_tuner.HyperModel):
    def build(self, hp) -> Model:
        return