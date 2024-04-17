"""
Run this module to act as a worker for distributed tuning. Change num_workers based on your
hardware limitations. Change line 8 to the root directory of the 208-Project on your machine.
Just terminate the file to pause the training
"""

import sys

# Needed for VSCode. Replace with path to 208-project on your machine
sys.path.append("C:/Users/steph/PycharmProjects/208-Project")

import os.path
import json
import keras_tuner
import pika
from Final_Version.Models import HyperCGAN
from Final_Version import Data_Handler
import Distributed_Tuner

FILE_PATH = "./tasks.json"
MAX_WORKERS = 1


def remove_task(trial_id):
    """
    Removes a task from the FILE_PATH json file

    :param trial_id: The trial to be removed
    """
    tasks = load_tasks()
    tasks.pop(f"{trial_id}")
    with open(FILE_PATH, "w") as file:
        json.dump(tasks, file)


def load_tasks():
    """
    Loads and converts to a dict the FILE_PATH json file

    :return: The loaded dict of trial ids to hyperparameters
    """
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            return json.load(file)
    else:
        return {}


def run_trial(task):
    """
    Runs a trial for a given task

    Given a task it will run the trial using the distributed tuner.
    Once the search has finished it will remove the FILE_PATH json file

    :param task: A Tuple with (trial_id, dict of hyperparameters)
    """
    task_id, hyperparameters = task
    hp = keras_tuner.HyperParameters()
    for k, v in hyperparameters.items():
        hp.Choice(k, [v])
    images, labels = Data_Handler.load_test_dataset()

    print(f"Starting trial {task_id}")
    tuner = Distributed_Tuner.Distributed_Tuner(
        hypermodel=HyperCGAN.HyperCGAN(),
        directory="hyper_tuning",
        objective=keras_tuner.Objective("Discriminator Loss", "min"),
        project_name='MyTuner',
        hyperparameters=hp,
        overwrite=True,
        trial_id=f"{task_id}",

    )
    tuner.search(images, labels, epochs=500)
    remove_task(task_id)


def save_task(info, trial_id):
    """
    Saves the task to the FILE_PATH json file

    :param info: A dict of hyperparameters to a value
    :param trial_id: The trial_id corresponding to the set of hyperparameters
    """
    tasks = load_tasks()
    info["running"] = True
    tasks[trial_id] = info
    with open(FILE_PATH, "w") as file:
        json.dump(tasks, file)


def callback(ch, method, _, body):
    """
    Callback method for what to do when receiving a message

    The method acknowledges the message and closes the connection and then
    adds it to the tasks in the FILE_PATH json file. Finally, it calls
    run_trial to start the tuning of this message

    :param ch: The pika channel the message came from
    :param method: Information about the message used to acknowledge it
    :param _: Unused in this method
    :param body: The contents of the method
    """
    ch.basic_ack(delivery_tag=method.delivery_tag)
    ch.close()
    info = json.loads(body)
    trial_id = info["trial_id"]
    info.pop("trial_id")
    save_task(info, trial_id)
    run_trial((trial_id, info))


def run_a_thread():
    """
    Constantly checks for a new task. Once one is received, it calls the method
    callback in order to begin the tuning for that message
    """
    while True:
        connection = pika.BlockingConnection(pika.URLParameters(
            'amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
        channel = connection.channel()

        channel.queue_declare(queue='Tuning', durable=True)
        print('Waiting for tasks')

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue='Tuning', on_message_callback=callback)

        channel.start_consuming()


if __name__ == "__main__":
    trial_id = 77
    info = {"Generator LR": 0.0001, "Discriminator LR": 0.0001, "Batch Size": 32, "Latent Dim": 128}
    run_trial((trial_id, info))
