import paho.mqtt.client as mqtt
from time import sleep

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: " + str(rc))
    client.subscribe("hello/#")
    print("Listening")


def on_message(client, userdata, message):
    print("Received message " + str(message.payload.decode('utf-8')))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set("colin","cpasswd1")
client.connect("localhost", 1883, 60)
client.loop_forever()


