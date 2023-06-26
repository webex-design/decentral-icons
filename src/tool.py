import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image

regFilter = re.compile(r'.(jpg|png)$', re.I)
current_file_path = os.path.realpath(__file__)
current_file_dir = os.path.dirname(current_file_path)

output_path = os.path.abspath(os.path.join(current_file_dir, '../data/3.jpg'))
style_path = os.path.abspath(os.path.join(current_file_dir, '../data/styles/0.jpg'))
content_path = os.path.abspath(os.path.join(current_file_dir, '../data/photo/1.png'))

smodel = os.path.abspath(os.path.join(current_file_dir, '../data/models/magenta_arbitrary-image-stylization-v1-256_2'))
hub_module = hub.load(smodel)

super_model = os.path.abspath(os.path.join(current_file_dir, '../data/models/esrgan-tf2_1'))
hub_super_model = hub.load(super_model)

def loadImage(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    output = tf.image.resize(img, [32,32])
    img = tf.image.resize(img, [256,256])
    img = img[tf.newaxis, :]
    return img

def predict_img(content, style):
    outputs = hub_module(content, style)
    stylized_image = tf.image.resize(outputs[0], [32,32])
    t = stylized_image * 256
    t = np.array(t, dtype=np.uint8)[0]
    return t

def run():
    generated_image = hub_module(loadImage(content_path), loadImage(style_path))
    stylized_image = tf.image.resize(generated_image[0], [230,413])
    # stylized_image = hub_super_model(stylized_image)
    tf.keras.preprocessing.image.save_img(output_path, stylized_image[0])
    
    
with tf.device('/GPU:0'):
    run()