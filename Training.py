#things I am importing because I might need them
import torch.nn as nn
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import datetime
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
from matplotlib import pyplot
from matplotlib import gridspec

from Models.Generator import Generator
from Models.Discriminator import Discriminator
import DataHandler
import evaluator
import numpy

#Method parameters not stubbed as dependant on implementation
#image input may need normalising to (128,128,3) as part of the dataloader
def train_one_epoch(dataset) -> None:
    for batch,labels in dataset:
        random_image_noise = numpy.random.rand(128,128,128,3) #I am assuming a batch size of 128
        random_image_noise = convert_to_tensor(random_image_noise, dtype=float32)
        fake_images = generator([random_image_noise,labels])

        # usually one would use .compile() and .fit() for this purpose, however since our model has its target labels concatenated into its input
        # this standard method not work and we must calculate the variance of output from expected using like_one or like_zero on the output labels
        # then calculate gradient and store this on gradient tape,
        # before computing loss by running on a manually instantiated optimizer.
        # train discriminator on real images
        
        # train discriminator on fake images
        # train generator on noise
    #TODO: Implement training for both nets for one epoch
    return

def train(dataset,5) -> None:
    for epoch in range(1,epochs):
        train_one_epoch(dataset)
    #TODO: Implement training loops to call train_one_epoch and do any needed setup
    return


if __name__ == "__main__":
    print("hi")
    train(ds,5)
    #TODO: Call methods to setup and begin training. This is equivilant of the main method in java
    #If statement used to dictate only main thread can execute not worker threads
