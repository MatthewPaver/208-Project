# README

This README provides a guide on how to run Worker.py to tune different sets of hyperparameters

## **Download All Dependencies.**
```bash
pip install keras_tuner
pip install pika
```
Also visit https://www.tensorflow.org/install/pip#cpu and download tensorflow for your platform.

## **Determining amount of threads to run**
Worker.py waits for a message from a cloud message broker, decodes it and runs the trial with
the parameters in the message. It is capable of running multiple instances at once with the variable
NUM_OF_WORKERS setting the maximum amount of threads it can run. Run the file with just one worker
as it's set to by default and increase it if you can. The major limiting factor is RAM so if you notice
RAM is nearing ~80-90 percent do not increase it further.

## **Running as a worker**
After determining how many threads you can run, you can leave Worker.py running and it will continuously
run trials. This can be paused by terminating the program and it will reset to the last saved epoch. 