import tensorflow as tf
import os
import math
import shutil
import numpy as np
from my_data_pair import MyData

class Combine:
    def __init__(self, root, channels=3):
        self.channels = channels
        self.root = root
        shutil.rmtree(root)
        if not os.path.exists(root):
            os.makedirs(root)
        
    def combine(self, source_path, config_path, list, presentor, key):
        my_data = MyData(self.channels)
        my_data.read_list(source_path, config_path, list)
        dataset = my_data.create_dataset()
        total = my_data.num_label
        side_length = int(math.sqrt(total))
        rows = math.ceil( total / side_length)
        toadd = side_length * rows - total
        list = []
        for step, (input_image, labels) in dataset.take(total).enumerate():
            t = (input_image + 1 ) * 127.5
            if(tf.equal(labels, presentor)):
                n,height, width, channels = tf.shape(t)
                mask = tf.pad(tf.ones([height - 2, width - 2]), [[1, 1], [1, 1]], constant_values=False)
                mask_expanded = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.bool)
                mask_expanded = tf.tile(mask_expanded, [1, 1, channels])
                # 创建红色像素张量
                red_color = tf.constant([255, 0, 0], dtype=tf.float32)
                # 将最外边缘像素设置为红色
                t = tf.where(mask_expanded, t, red_color)
            list.append(t[0])
        list_outs = self.combine_predict(list, rows, toadd)
        tf.keras.preprocessing.image.save_img(os.path.join(self.root , f'{key}.jpg'), list_outs)
        return
    
    def combine_predict(self, list, rows, toadd):
        for i in range(toadd):
            list.append(tf.fill((32, 32, self.channels),  255))
        
        lists = np.array_split(list, rows)
        tflist = []
        for element in lists:
            newlist = tf.concat(element, axis=0)
            newlist = tf.reshape(newlist, (32 * len(element), 32, self.channels))
            tflist.append(newlist)       
        concatenated_tensor = tf.concat(tflist, axis=1)
        return concatenated_tensor
