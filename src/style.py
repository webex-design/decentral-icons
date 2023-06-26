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
source_regular = os.path.abspath(os.path.join(current_file_dir, '../data/source/regular'))
style_dir = os.path.abspath(os.path.join(current_file_dir, f'../data/styles/'))
style_paths = [os.path.join(style_dir,p) for p in os.listdir(style_dir)]
style_paths = [p for p in style_paths if os.path.isfile(p) and re.search(regFilter, p)!=None]
style_num = len(style_paths)

content_files= ['1386@1885.jpg','1386@1937.jpg']
content_paths = [os.path.join(source_regular, f) for f in content_files]

smodel = os.path.abspath(os.path.join(current_file_dir, '../data/models/magenta_arbitrary-image-stylization-v1-256_2'))
hub_module = hub.load(smodel)

def loadImage(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    output = tf.image.resize(img, [32,32])
    img = tf.image.resize(img, [256,256])
    img = img[tf.newaxis, :]
    return [img,output]

def predict_img(content, style):
    outputs = hub_module(content, style)
    stylized_image = tf.image.resize(outputs[0], [32,32])
    t = stylized_image * 256
    t = np.array(t, dtype=np.uint8)[0]
    return t

def run():
    contents_source = [loadImage(p) for p in content_paths]
    contents_tf = [tf.constant(item[0]) for item in contents_source]

    style_images_source = [loadImage(p) for p in style_paths ]
    style_images = [tf.constant(item[0]) for item in style_images_source ]

    source_images = [t[1] for t in style_images_source]

    images_to_show = [source_images]
    for content_tf in contents_tf:
        images_to_show.append([predict_img(content_tf, i) for i in style_images])

    # fig_width, fig_height = 32, 32  # 单位为像素
    # dpi = 72  # 每英寸的像素数
    # fig_size = (fig_width / dpi, fig_height / dpi)
    lenN = len(images_to_show)
    fig, axes = plt.subplots(nrows=lenN, ncols=style_num)    
    
    for i in range(lenN):
        for j in range(style_num):
            # axes[i][j].imshow(images_to_show[i][j])  # 绘制图片
            # axes[i][j].axis('off')
            axes.flat[i * style_num + j].imshow(images_to_show[i][j])  # 绘制图片
            axes.flat[i * style_num + j].axis('off')
    plt.tight_layout()
    plt.show()
    
with tf.device('/GPU:0'):
    run()