# README

This README provides a guide on how to run Worker.py to tune different sets of hyperparameters

## **Download All Dependencies.** ##
```bash
pip install keras_tuner
pip install pika
pip install matplotlib
pip install tensorflow-datasets
```
Also visit https://www.tensorflow.org/install/pip#cpu and download tensorflow for your platform.
For windows the command is:
```bash
pip install tensorflow
```

## **Quick Overview** ##
This will be a quick run-down of what worker.py does if you don't want
to know you're free to skip this section. 

Hypertuning is just training a model multiple times with different settings (also called hyperparameters) and then picking
the best set of these for your final model. I have sent some messages to a cloud server which corresponds to 
a set of these hyperparameters. Your computers will pull down one message and fully train the model using these parameters.
Each instance of doing this is known as a trial

## **Actually running trials** ##
After downloading all dependencies, Worker.py in Final_version>Distributed_HyperTuning>Worker.py has been designed
for you to run it and not really have to think about it again. It will automatically pull down a trial and
start running that trial. Each trial will run for 200 epochs and it will automatically pick up a new trial once
it's finished. 

Worker.py is also designed to save the progress of the training each epoch. The files for this can be found in
Final_version>Distributed_HyperTuning>hyper_tuning>MyTuner>trial_{trialID} where the trialID is the ID used to identify
that trial. This means that the python script can be terminated at any time with a maximum time loss of about 20 minutes

Also saved in that directory is an image file containing a grid of 9 seperate images.
These correlate with the 9 different classes the CGAN can generate and can be used to see how well your model is learning

tasks.json is also created in the Distributed_HyperTuning directory which helps with reloading the trial after a termination

It is important all these files are not deleted. None of these should be uploaded to GitHub either until I tell you which 
ones to upload. 

# **DO NOT RUN Task_Setter.py**