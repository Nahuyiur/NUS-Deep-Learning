import paho.mqtt.client as mqtt
import base64
import numpy as np
import cv2
import json
from classify.predict_breed import predict_cat  # 你之前写的模块
from tensorflow.keras.models import load_model

# 配置
MQTT_BROKER = 'localhost'
TOPIC_CLASSIFY = 'Group_01/IMAGE/classify'
MODEL_PATH = '/path/to/your_model.h5'
CLASS_NAMES = ['Pallas cats', 'Persian cats', 'Ragdolls', 'Singapura cats', 'Sphynx cats']

# 预加载模型
model = load_model(MODEL_PATH, compile=False)

def on_connect(client, userdata, flags, rc):
    print(f'Connected with result code {rc}')
    client.subscribe(TOPIC_CLASSIFY)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        img_b64 = payload['image']
        img_bytes = base64.b64decode(img_b64)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # 保存原图
        cv2.imwrite('received.jpg', img)

        # 分类预测
        from tensorflow.keras.applications.efficientnet import preprocess_input
        from tensorflow.keras.preprocessing import image
        img_resized = cv2.resize(img, (300, 300))
        arr = preprocess_input(img_resized.astype('float32'))
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        conf = preds[idx]

        print(f'✅ Predicted: {label} ({conf:.4f})')

        # 显示图像 + 结果（可选）
        cv2.putText(img, f'{label} ({conf:.2f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Received & Classified", img)
        cv2.waitKey(1)

    except Exception as e:
        print(f'❌ Error: {e}')

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, 1883, 60)
client.loop_forever()
