import os
import re
import tensorflow as tf
from tensorflow import keras

default_re = re.compile(r'.(jpg|png)$', re.I)

class Datas:
    def __init__(self, image_shape = (32, 32, 1), batch_size = 64):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.dataset_repetitions = 3
        return
    
    def __scan(self, path, filter=default_re):
        files = os.listdir(path)
        ret = []
        for x in files:
            _path = os.path.join(path, x)
            if os.path.isdir(_path):
                ret += self.__scan(path)
            elif filter==None or re.search(filter, x)!=None:
                ret.append(_path)
        return ret
    
    def __read_input_images(self, path):
        input_images = {}
        for x in os.listdir(path):
            folder = os.path.join(path, x)
            if os.path.isdir(folder):
                input_images[x] = self.__scan(folder)
        return input_images
    
    # Normalizing the images [0,255] to [-1, 1]
    def __normalize(self, image):
        return image / 255
    
    def __random_crop(self, image):
        _image = tf.image.resize(image, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.image.random_crop(_image, size=list(self.image_shape))
    
    def __load(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, self.image_shape[2])
        image = tf.image.resize(image, list(self.image_shape[:2]))
        return tf.cast(image, tf.float32)
    
    @tf.autograph.experimental.do_not_convert
    def load_train(self, input_path):
        input_image = self.__load(input_path)
        input_image = self.__normalize(input_image)
        return input_image

    @tf.autograph.experimental.do_not_convert
    def load_crop(self, input_path):
        input_image = self.__load(input_path)
        input_image = self.__random_crop(input_image)
        input_image = self.__normalize(input_image)
        return input_image
    
    def read(self, source_path):
        source_images = []
        handdrawn_images = []
        labels = []
        ipts = self.__read_input_images(source_path)
        for key, value in ipts.items():
            if len(value) > 1:
                source_images.append(os.path.join(source_path, key, '0.jpg'))
                for _path in value:
                    if os.path.basename(_path) != '0.jpg':
                        handdrawn_images.append(_path)
                        labels.append(int(key))
                        break
        self.source = (source_images, handdrawn_images, labels)
        return self.source
    
    def read_max(self, source_path, max = 1):
        source_images = []
        handdrawn_images = []
        labels = []
        ipts = self.__read_input_images(source_path)
        sorted_items = sorted(ipts.items(), key=lambda x: len(x[1]), reverse=True)
        for key, value in sorted_items[:max]:
            for _path in value:
                source_images.append(_path)
                handdrawn_images.append(_path)
                labels.append(int(key))
        self.source = (source_images, handdrawn_images, labels)
        return self.source
    
    def read_folder(self, source_path, max = 1):
        source_images = []
        handdrawn_images = []
        labels = []
        ipts = self.__scan(source_path)
        for _path in ipts:
            source_images.append(_path)
            handdrawn_images.append(_path)
            labels.append(0)
        self.source = (source_images, handdrawn_images, labels)
        return self.source

    def read_all(self, source_path):
        source_images = []
        handdrawn_images = []
        labels = []
        ipts = self.__read_input_images(source_path)
        for key, value in ipts.items():
            source_images.append(os.path.join(source_path, key, '0.jpg'))
            if len(value) > 1:
                for _path in value:
                    if os.path.basename(_path) != '0.jpg':
                        handdrawn_images.append(_path)
                        labels.append(int(key))
        self.source = (source_images, handdrawn_images, labels)
        return self.source
        
    def create_dataset_train(self):
        (source_images, handdrawn_images, labels) = self.source
        return self.create_dataset_one(source_images, self.load_train, 0.7)
    
    def create_dataset_test(self):
        (source_images, handdrawn_images, labels) = self.source
        return self.create_dataset_one(source_images, self.load_train)
    
    def create_dataset_valid(self):
        (source_images, handdrawn_images, labels) = self.source
        return self.create_dataset_one(source_images, self.load_train, -0.7)
    
    def create_dataset_one(self, sources, load_func, ratio):
        split_index = int(len(sources) * abs(ratio))
        realsource = sources[:split_index] if ratio > 0 else sources[split_index:]
        dataset_train = tf.data.Dataset.from_tensor_slices(realsource)
        dataset_train = dataset_train.map(load_func).cache()
        dataset_train = dataset_train.repeat(self.dataset_repetitions)
        dataset_train = dataset_train.shuffle(buffer_size=len(realsource))
        dataset_train = dataset_train.batch(batch_size=self.batch_size, drop_remainder=True)
        dataset_train = dataset_train.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset_train
