import pika
import json

connection = pika.BlockingConnection(pika.URLParameters('amqps://bfjzexuw:h91qsaFYNrHc8Ag_5WVOOVdFH2MpnOby@whale.rmq.cloudamqp.com/bfjzexuw'))
channel = connection.channel()

channel.queue_declare(queue='Tuning', durable=True)

hp = {"lr1": 0.1, "lr2": 0.1, "batch_size": 128, "trial_id": "Trial_00"}
message = json.dumps(hp)

channel.basic_publish(exchange='', routing_key='Tuning', body=str(message), properties=pika.BasicProperties(delivery_mode=2))
print(" [x] Sent task:", message)

connection.close()