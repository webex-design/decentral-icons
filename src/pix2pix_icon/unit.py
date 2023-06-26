import tensorflow as tf
from tensorflow import keras

class ConvBNRelu(keras.Model):
    def __init__(self, filters, kernel=3, strides=2, padding='same', use_bn=True):
        super(ConvBNRelu, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.use_bn = use_bn
        self.conv = keras.layers.Conv2D(filters, kernel, strides, padding, kernel_initializer=initializer)
        self.relu = keras.layers.LeakyReLU()
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
    def __init__(self, filters, kernel=5, strides=2, padding='same', use_drop=False):
        super(DeConvBNRelu, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.use_drop = use_drop
        # valid => N2 = (N-1) * strides + KernelSize
        # same => N2 = N * strides
        self.conv = keras.layers.Conv2DTranspose(filters, kernel, strides, padding, kernel_initializer=initializer)
        self.relu = keras.layers.LeakyReLU()
        self.bn = tf.keras.layers.BatchNormalization()
        self.drop = None
        if use_drop:
            self.drop = tf.keras.layers.Dropout(0.25)
    
    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.use_drop:
            x = self.drop(x)
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
    
class DeResBlock(keras.Model):
    def __init__(self, filters, kernel=3, strides=1, padding='same', use_drop=False):
        super(DeResBlock, self).__init__()
        self.conv1 = DeConvBNRelu(filters, kernel, strides, padding, use_drop)
        self.conv2 = DeConvBNRelu(filters, kernel, strides, padding, use_drop)
       
    @tf.function 
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        return x