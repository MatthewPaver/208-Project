from Models.Generator import Generator
from Models.Discriminator import Discriminator
import DataHandler
import evaluator

#Method parameters not stubbed as dependant on implementation

def train_one_epoch() -> None:
    #TODO: Implement training for both nets for one epoch
    return

def train(dataset,5) -> None:
    #TODO: Implement training loops to call train_one_epoch and do any needed setup
    return


if __name__ == "__main__":
    print("hi")
    train(ds,5)
    #TODO: Call methods to setup and begin training. This is equivilant of the main method in java
    #If statement used to dictate only main thread can execute not worker threads
