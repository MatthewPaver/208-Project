import pika
import json
from itertools import product

FILE_PATH = "../AllTasks.json"

def set_tasks():
    connection = pika.BlockingConnection(pika.URLParameters('amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
    channel = connection.channel()

    channel.queue_declare(queue='Tuning', durable=True)

    with open(FILE_PATH, "r") as file:
        tasks_to_be_set = json.load(file)

    for i in tasks_to_be_set:
        message = json.dumps(i)
        channel.basic_publish(exchange='', routing_key='Tuning', body=str(message),properties=pika.BasicProperties(delivery_mode=2))
        print("Sent task:", message)

    connection.close()

def create_tasks():
    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    batch_size = [4,8]
    latent_dim = [100]
    id = 0
    combinations = product(lr, lr, batch_size, latent_dim)
    list_of_trials = []
    for i in combinations:
        trial = {'Generator LR': i[0], 'Discriminator LR': i[1], "Batch Size": i[2], "Latent Dim": i[3], "trial_id": id}
        list_of_trials.append(trial)
        id += 1
    with open(FILE_PATH, "w") as file:
        json.dump(list_of_trials, file)

set_tasks()