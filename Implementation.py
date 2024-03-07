from keras_tuner import tuners
from Data_Handler import load_dataset
from Models.HyperCGAN import HyperCGAN


#TODO: Change objective to be a tracked metric

if __name__ == "__main__":
    x , y = load_dataset()
    tuner = tuners.GridSearch(
        hypermodel=HyperCGAN(),
        overwrite=True,
        project_name="hyper_tuning"
    )

    tuner.search(x,y, epochs=5)
