import os
import re
import json
import tensorflow as tf

default_re = re.compile(r'.(jpg|png)$', re.I)

class Datas:
    def __init__(self, image_shape = (32, 32, 1)):
        self.image_shape = image_shape
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
    
    def __read_config(self, path):
        with open(path, 'r') as f:
            json_data = f.read()
        return json.loads(json_data)
    
    # Normalizing the images [0,255] to [-1, 1]
    def __normalize(self, image):
        return (image / 127.5) - 1
    
    def __random_crop(self, image):
        _image = tf.image.resize(image, [50, 50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.image.random_crop(_image, size=list(self.image_shape))
    
    def __load(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, self.image_shape[2])
        image = tf.image.resize(image, list(self.image_shape[:2]))
        return tf.cast(image, tf.float32)
    
    def __load_image(self, input_path, real_path):
        input_image = self.__load(input_path)
        real_image = self.__load(real_path)
        return input_image, real_image

    @tf.autograph.experimental.do_not_convert
    def load_test(self, input_path, real_path, labels):
        input_image, real_image = self.__load_image(input_path, real_path)
        input_image = self.__random_crop(input_image)
        real_image = self.__random_crop(real_image)
        input_image = self.__normalize(input_image)
        real_image = self.__normalize(real_image)
        return input_image, real_image, labels
    
    @tf.autograph.experimental.do_not_convert
    def load_train(self, input_path, real_path, labels):
        input_image, real_image = self.__load_image(input_path, real_path)
        input_image = self.__normalize(input_image)
        real_image = self.__normalize(real_image)
        return input_image, real_image, labels
    
    @tf.autograph.experimental.do_not_convert
    def preprocess_image_train(self, path):
        image = self.__load(path)
        image = self.__normalize(image)
        return image
    
    @tf.autograph.experimental.do_not_convert
    def preprocess_image_test(self, path):
        image = self.__load(path)
        image = self.__random_crop(image)
        image = self.__normalize(image)
        return image
    
    def read(self, source_path , config_path):
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
        self.config = self.__read_config(config_path)
        self.source = (source_images, handdrawn_images, labels)
        return self.source

    def read_all(self, source_path , config_path):
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

        self.config = self.__read_config(config_path)
        self.source = (source_images, handdrawn_images, labels)
        return self.source
        
    def create_dataset(self):
        source_images, handdrawn_images, labels = self.source
        source = (source_images, handdrawn_images, labels)
        self.num_label = len(source_images)
        
        dataset_train = tf.data.Dataset.from_tensor_slices(source)
        dataset_train = dataset_train.map(self.load_train)
        dataset_train = dataset_train.shuffle(buffer_size=self.num_label)
        dataset_train = dataset_train.batch(batch_size=1)
        
        return dataset_train
    
    def create_dataset_one(self, sources):
        dataset_train = tf.data.Dataset.from_tensor_slices(sources)
        dataset_train = dataset_train.map(self.preprocess_image_train)
        dataset_train = dataset_train.shuffle(buffer_size=len(sources))
        dataset_train = dataset_train.batch(batch_size=1)
        return dataset_train
    
    
    def create_dataset_and_test(self, BATCH_SIZE=1):
        source_images, handdrawn_images, labels = self.source
        BUFFER_SIZE = self.num_label = len(source_images)
        AUTOTUNE = tf.data.AUTOTUNE
        dataset_source = tf.data.Dataset.from_tensor_slices(source_images)
        dataset_target = tf.data.Dataset.from_tensor_slices(handdrawn_images)
        
        dataset_source = dataset_source.cache().repeat().map(
            self.preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

        dataset_target = dataset_target.cache().repeat().map(
            self.preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

        dataset_source_test = dataset_source.map(
            self.preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

        dataset_source_test = dataset_target.map(
            self.preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

        return dataset_source, dataset_target, dataset_source_test, dataset_source_test
    