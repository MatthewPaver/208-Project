import pika
import json

connection = pika.BlockingConnection(pika.URLParameters('amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
channel = connection.channel()

channel.queue_declare(queue='Tuning', durable=True)

hp = {'Generator LR': 0.1, 'Discriminator LR': 0.1, "Batch Size": 128, "Latent Dim": 100, "trial_id": "Trial_00"}
message = json.dumps(hp)

channel.basic_publish(exchange='', routing_key='Tuning', body=str(message), properties=pika.BasicProperties(delivery_mode=2))
print("Sent task:", message)

connection.close()