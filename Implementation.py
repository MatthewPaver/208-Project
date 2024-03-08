from keras_tuner import tuners
from Data_Handler import load_dataset
from Models import HyperCGAN

if __name__ == "__main__":
    x , y = load_dataset()
    tuner = tuners.GridSearch(
        hypermodel=HyperCGAN.HyperCGAN(),
        overwrite=False,
        project_name="hyper_tuning"
    )

    tuner.search(x,y, epochs=5)
