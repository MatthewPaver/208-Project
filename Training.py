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
        random_image_noise = tf.convert_to_tensor(random_image_noise, dtype=float32)
        fake_images = generator([
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
