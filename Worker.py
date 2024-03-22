import os.path
import keras_tuner
import json
import Distributed_Tuner
import Data_Handler
from Models import HyperCGAN
from multiprocessing import Pool
import pika
from concurrent.futures import ThreadPoolExecutor


FILE_PATH = "tasks.json"


def remove_task(trial_id):
    tasks = load_tasks()
    tasks.pop(f"{trial_id}")
    with open(FILE_PATH, "w") as file:
        json.dump(tasks, file)

def load_tasks():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            return json.load(file)
    else:
        return {}


def run_trial(args):
    id, hyperparameters = args
    hp = keras_tuner.HyperParameters()
    for k, v in hyperparameters.items():
        hp.Choice(k, [v])
    x, y = Data_Handler.load_dataset()

    print(f"Starting trial {id}")
    tuner = tuner2.Distributed_Tuner(
        hypermodel= HyperCGAN.HyperCGAN(),
        directory= "hyper_tuning",
        objective= keras_tuner.Objective("Generator Loss", "min"),
        project_name='MyTuner',
        hyperparameters= hp,
        overwrite=False,
        trial_id=f"{id}",
    )
    tuner.search(x,y, epochs=2)
    remove_task(id)


def save_task(info, trial_id):
    tasks = load_tasks()
    info["running"] = True
    tasks[trial_id] = info
    with open(FILE_PATH, "w") as file:
        json.dump(tasks, file)

def callback(ch, method, _, body):
    ch.basic_ack(delivery_tag=method.delivery_tag)
    ch.close()
    info = json.loads(body)
    id = info["trial_id"]
    info.pop("trial_id")
    save_task(info,id)
    run_trial((id, info))

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
    tasks = load_tasks()
    max_workers = 2
    if tasks:
        with Pool(processes=2) as pool:
            pool.map(run_trial, tasks.items())
            pool.close()
            pool.join()
    print("tasks finished")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(max_workers):
            executor.submit(run_a_thread)