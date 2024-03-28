"""
This module creates tasks and then sends them off as messages to CloudAMQP
"""
import json
import pika

FILE_PATH = "../AllTasks.json"


def send_tasks(tasks_to_be_sent):
    """
    This function sends tasks to the cloud queue ready for any available workers to pull down
    """
    connection = pika.BlockingConnection(
        pika.URLParameters
        ('amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
    channel = connection.channel()

    channel.queue_declare(queue='Tuning', durable=True)

    for i in tasks_to_be_sent:
        message = json.dumps(i)
        channel.basic_publish(exchange='', routing_key='Tuning', body=str(message),
                              properties=pika.BasicProperties(delivery_mode=2))
        print("Sent task:", message)

    connection.close()


#Template
#{"Generator LR": 0.001, "Discriminator LR": 0.0001, "Batch Size": 128, "Latent Dim": 100, "trial_id": 1}

def create_new_tasks():
    """
    This function loads all old tasks and adds new_tasks to it. It then calls send_tasks to issue the
    new tasks to the cloud before writing all tasks to the file in FILE_PATH
    """
    with open(FILE_PATH, "r") as file:
        old_tasks = json.load(file)

    new_tasks = [{"Generator LR": 0.001, "Discriminator LR": 0.0001, "Batch Size": 128, "Latent Dim": 100, "trial_id": 1},
                 {"Generator LR": 0.001, "Discriminator LR": 0.001, "Batch Size": 128, "Latent Dim": 100, "trial_id": 2},
                 {"Generator LR": 0.01, "Discriminator LR": 0.001, "Batch Size": 128, "Latent Dim": 100,"trial_id": 3},
                 {"Generator LR": 0.01, "Discriminator LR": 0.01, "Batch Size": 128, "Latent Dim": 100, "trial_id": 4},
                 ]

    old_tasks.append(new_tasks)

    send_tasks(new_tasks)

    with open(FILE_PATH, "w") as file:
        json.dump(old_tasks, file)
