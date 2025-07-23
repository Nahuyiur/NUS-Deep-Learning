import cv2
import base64
import json
import paho.mqtt.client as mqtt

img_path = 'test_cat.jpg'  # 可替换为相机拍摄
img = cv2.imread(img_path)
_, buffer = cv2.imencode('.jpg', img)
img_b64 = base64.b64encode(buffer).decode('utf-8')

payload = json.dumps({'image': img_b64})

client = mqtt.Client()
client.connect('电脑IP', 1883, 60)
client.publish('Group_01/IMAGE/classify', payload)
client.disconnect()
