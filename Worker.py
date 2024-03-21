import os.path
import pika
import Tuner
import keras_tuner
import json
import Data_Handler
from Models import HyperCGAN

FILE_PATH = "Task.json"

def callback(ch, method, _, body):
    ch.basic_ack(delivery_tag=method.delivery_tag)
    dict = json.loads(body)
    with open(FILE_PATH, "w") as file:
        json.dump(dict, file)
    run_trial(dict)


def run_trial(hyperparameters):
    hp = keras_tuner.HyperParameters()
    for k, v in hyperparameters.items():
        hp.Choice(k, [v])

    x, y = Data_Handler.load_dataset()

    tuner = Tuner.MyTuner(
        hypermodel= HyperCGAN.HyperCGAN(),
        directory= "hyper_tuning",
        objective= keras_tuner.Objective("Generator Loss", "min"),
        project_name='MyTuner1',
        allow_new_entries= False,
        hyperparameters= hp,
        overwrite=False,
        trial_id="MyTestTrial",
    )
    tuner.search(x,y, epochs=2)
    os.remove(FILE_PATH)


#--------------------------------------------------Main-----------------------------------------------------------------

if os.path.exists(FILE_PATH):
    with open(FILE_PATH, "r") as file:
        data = json.load(file)
    run_trial(data)
else:
    connection = pika.BlockingConnection(pika.URLParameters('amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
    channel = connection.channel()

    channel.queue_declare(queue='Tuning', durable=True)
    print('Waiting for tasks')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='Tuning', on_message_callback=callback)

    channel.start_consuming()
