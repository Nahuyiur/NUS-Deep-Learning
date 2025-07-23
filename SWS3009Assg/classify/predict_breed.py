import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D

# compat layer: ignore 'groups' arg
class DepthwiseConv2DCompat(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kw):
        super().__init__(*args, **kw)

IMG_SIZE = 300

def _prep(img_path: str, size=IMG_SIZE):
    img = image.load_img(img_path, target_size=(size, size))
    arr = preprocess_input(image.img_to_array(img))
    return tf.expand_dims(arr, 0)

def predict_cat(model_path: str, img_path: str, classes: list[str]):
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={'DepthwiseConv2D': DepthwiseConv2DCompat}
    )
    probs = model.predict(_prep(img_path), verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])