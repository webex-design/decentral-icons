import numpy as np
import tensorflow as tf
from tensorflow import keras
from cdata import CData

class ConvBNRelu(keras.Model):
    def __init__(self, filters, kernel=3, strides=1, padding='same', use_bn=True):
        super(ConvBNRelu, self).__init__()
        self.use_bn = use_bn
        self.conv = keras.layers.Conv2D(filters, kernel, strides, padding)
        self.relu = keras.layers.ReLU()
        self.bn = None
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization()
    
    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x
    
class DeConvBNRelu(keras.Model):
    def __init__(self, filters, kernel=5, strides=2, padding='same', use_bn=True):
        super(DeConvBNRelu, self).__init__()
        self.use_bn = use_bn
        # valid => N2 = (N-1) * strides + KernelSize
        # same => N2 = N * strides
        self.conv = keras.layers.Conv2DTranspose(filters, kernel, strides, padding)
        self.relu = keras.layers.ReLU()
        self.bn = None
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization()
    
    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x
    
class ResBlock(keras.Model):
    def __init__(self, filters, kernel=3, strides=1, padding='same', use_bn=True):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBNRelu(filters, kernel, strides, padding, use_bn)
        self.conv2 = ConvBNRelu(filters, kernel, strides, padding, use_bn)
       
    @tf.function 
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        return x

class Cnn:
    def __init__(self, data_dir):
        cdata = CData()
        mydict = cdata.read_folder(data_dir)
        self.num_label = len(mydict.keys())
        self.source = cdata.from_dict(mydict)
        return
    
    def __load_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)
        return image, label
    
    def input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.source)
        dataset = dataset.map(self.__load_image)
        dataset = dataset.shuffle(buffer_size=self.num_label)
        dataset = dataset.batch(batch_size=120)
        self.dataset = dataset
        return dataset
    
    def model_fn(self):
        inputs = tf.keras.layers.Input(shape=(32, 32, 1))
        
        res0 = tf.keras.Sequential([ResBlock(64) for i in range(2)])
        cnn1 = ConvBNRelu(32, kernel=5,strides=2)
        res1 = tf.keras.Sequential([ResBlock(32) for i in range(1)])
        cnn2 = ConvBNRelu(32, kernel=5,strides=2)
        res2 = tf.keras.Sequential([ResBlock(32) for i in range(1)])
        flat = tf.keras.layers.Flatten()
        fc1 = tf.keras.layers.Dense(self.num_label*2, activation='relu')
        # drop1 = tf.keras.layers.Dropout(0.25)
        fc2 = tf.keras.layers.Dense(self.num_label, activation=None)
        
        x = res0(inputs)
        x = cnn1(x)
        x = res1(x)
        x = cnn2(x)
        x = res2(x)
        x = flat(x)
        x = fc1(x)
        # x = drop1(x)
        outputs = fc2(x)
        
        self.model =  tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    def fit(self, epochs=10):
        self.model.fit(self.dataset, epochs=epochs)
        
    def save(self, path):
        self.model.save(path)
        return

