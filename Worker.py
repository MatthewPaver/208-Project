import os.path
import pika
import Tuner
import keras_tuner
import json
import Data_Handler
from Models import HyperCGAN
from concurrent.futures import ThreadPoolExecutor

FILE_PATH = "tasks.json"
LOCKED_PATH = os.path.join(FILE_PATH, "LOCKED.txt")

def callback(ch, method, _, body):
    ch.basic_ack(delivery_tag=method.delivery_tag)
    ch.close()
    info = json.loads(body)
    id = info["trial_id"]
    info.pop("trial_id")
    save_task(info,id)
    run_trial(info, id)

def remove_task(trial_id):
    tasks = load_tasks()
    tasks.pop(trial_id)
    with open(FILE_PATH, "w") as file:
        json.dump(tasks, file)

def load_tasks():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            return json.load(file)
    else:
        return {}

def save_task(info, trial_id):
    tasks = load_tasks()
    tasks[trial_id] = info
    with open(FILE_PATH, "w") as file:
        json.dump(tasks, file)


def run_trial(hyperparameters, id):
    hp = keras_tuner.HyperParameters()
    for k, v in hyperparameters.items():
        hp.Choice(k, [v])
    x, y = Data_Handler.load_dataset()

    print(f"Starting trial {id}")
    tuner = Tuner.MyTuner(
        hypermodel= HyperCGAN.HyperCGAN(),
        directory= "hyper_tuning",
        objective= keras_tuner.Objective("Generator Loss", "min"),
        project_name='MyTuner',
        allow_new_entries= False,
        hyperparameters= hp,
        overwrite=False,
        trial_id=f"{id}",
    )
    tuner.search(x,y, epochs=2)
    remove_task(id)

def run_a_thread():
    while True:
        connection = pika.BlockingConnection(pika.URLParameters('amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
        channel = connection.channel()

        channel.queue_declare(queue='Tuning', durable=True)
        print('Waiting for tasks')

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue='Tuning', on_message_callback=callback)

        channel.start_consuming()

#--------------------------------------------------Main-----------------------------------------------------------------
if __name__ == "__main__":
    num_workers = 2
    tasks = load_tasks()
    if tasks:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for key in tasks.keys():
                ex.submit(run_trial, tasks[key], key)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for _ in range(num_workers):
                ex.submit(run_a_thread)