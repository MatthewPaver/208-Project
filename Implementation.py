"""
This module is the script to run to start the tuning process
"""
import keras_tuner

import Tuner
from Data_Handler import load_dataset
from Models import HyperCGAN


if __name__ == "__main__":
    x, y = load_dataset()
    tuner = Tuner.MyTuner(
        hypermodel=HyperCGAN.HyperCGAN(),
        objective= keras_tuner.Objective("g_loss", "max"),
        overwrite=False,
        project_name="hyper_tuning"
    )

    tuner.search(x,y, epochs=2)
