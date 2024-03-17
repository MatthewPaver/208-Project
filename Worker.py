import pika
import Tuner
import keras_tuner
import json
import Data_Handler
from Models import HyperCGAN

def callback(ch, method, properties, body):
    hyperparameters = json.loads(body)
    hp = keras_tuner.HyperParameters()
    for k, v in hyperparameters.items():
        hp.Choice(k, [v])

    x, y = Data_Handler.load_dataset()

    tuner = Tuner.MyTuner(
        hypermodel= HyperCGAN.HyperCGAN(),
        directory= "hyper_tuning",
        objective= keras_tuner.Objective("g_loss", "min"),
        project_name='MyTuner1',
        allow_new_entries= False,
        hyperparameters= hp,
        overwrite=False,
        trial_id="MyTestTrial",
    )
    tuner.search(x,y, epochs=2)


#--------------------------------------------------Main-----------------------------------------------------------------

connection = pika.BlockingConnection(pika.URLParameters('amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
channel = connection.channel()

channel.queue_declare(queue='task_queue', durable=True)
print(' [*] Waiting for tasks. To exit press CTRL+C')

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='task_queue', on_message_callback=callback)

channel.start_consuming()