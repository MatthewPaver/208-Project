from DataHandler import load_dataset
from keras_tuner import tuners
from Models.HyperCGAN import HyperCGAN


#TODO: Change objective to be a tracked metric

if __name__ == "__main__":
    x , y = load_dataset()
    tuner = tuners.GridSearch(
        hypermodel=HyperCGAN(),
    )


    tuner.search(x,y, epochs=5)

