import torch.nn as nn
from torch import Tensor
import numpy as np
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

#class Discriminator(nn.Module):
    #def __init__(self):
        #super(Discriminator, self).__init__()
        #TODO: Define all layers of the network
        #return

def image_preprocessor(generator_output+real):
    x = layers.Input(generator_output+real)
    return x #this is subject to change as I research how training works in tensorflow

def label_preprocessor(label_vector):
    x = layers.Input(label_vector)
    return x #this is subject to change as I research how training works in tensorflow

def create_discriminator():
    stream1_input = image_preprocessor()
    stream2_input = label_preprocessor()
    x = layers.Concatenate() ([stream1_input,stream2_input])
    
    x = layers.Conv2D(64,4) (x)
    x = layers.BatchNormalization() (x)
    x = layers.ReLu() (x)

    x = layers.Conv2D(64,4) (x)
    x = layers.BatchNormalization() (x)
    x = layers.ReLu() (x)

    x = layers.Conv2D(64,4) (x)
    x = layers.BatchNormalization() (x)
    x = layers.ReLu() (x)

    x = layers.Flatten() (x)
    x = layers.Dropout(0.3) (x)
    x = layers.Dense(1,activation ='sigmoid') (x)

    model = tf.keras.Model([stream1_input, stream2_input], x)

    return model

discriminator = create_discriminator():

    #def forward(self, input: Tensor) -> Tensor:
        #TODO: Define how the input passes through the layers
        #return
