from Models.Generator import Generator
from Models.Discriminator import Discriminator
import DataHandler

#Method parameters not stubbed as dependant on implementation

def train_one_epoch() -> None:
    #TODO: Implement training for both nets for one epoch
    return

def train() -> None:
    #TODO: Implement training loops to call train_one_epoch and do any needed setup
    return

def eval() -> float:
    #TODO: Implement a version of inception score to objectively determine performance of generator
    return

def weights_init(model: Generator) -> None:
    #TODO: Initialise the weights of the generator before training begins
    return

if __name__ == "__main__":
    print("hi")
    #TODO: Call methods to setup and begin training. This is equivilant of the main method in java
    #If statement used to dictate only main thread can execute not worker threads