import tensorflow as tf
from tensorflow import keras
# BatchNormalization LayerNormalization GroupNormalization UnitNormalization Normalization

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


class ConvBNRelu(keras.Model):
    def __init__(self, filters, kernel=3, strides=2, padding='same', normalization='InstanceNormalization'):
        super(ConvBNRelu, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv = keras.layers.Conv2D(filters, kernel, strides, padding, kernel_initializer=initializer)
        self.relu = keras.layers.LeakyReLU()
        self.normalization = None
        if normalization:
            self.normalization = InstanceNormalization() if normalization == 'InstanceNormalization' else tf.keras.layers.Normalization()
    
    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        if self.normalization:
            x = self.normalization(x)
        x = self.relu(x)
        return x
    
class DeConvBNRelu(keras.Model):
    def __init__(self, filters, kernel=5, strides=2, padding='same', use_drop=False, normalization='BatchNormalization'):
        super(DeConvBNRelu, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.use_drop = use_drop
        # valid => N2 = (N-1) * strides + KernelSize
        # same => N2 = N * strides
        self.conv = keras.layers.Conv2DTranspose(filters, kernel, strides, padding, kernel_initializer=initializer)
        self.relu = keras.layers.LeakyReLU()
        self.normalization = InstanceNormalization() if normalization == 'InstanceNormalization' else tf.keras.layers.Normalization()
        self.drop = None
        if use_drop:
            self.drop = tf.keras.layers.Dropout(0.25)
    
    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.normalization(x)
        if self.use_drop:
            x = self.drop(x)
        x = self.relu(x)
        return x

class ResBlock(keras.Model):
    def __init__(self, filters, kernel=3, strides=1, padding='same', normalization='BatchNormalization'):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBNRelu(filters, kernel, strides, padding, normalization)
        self.conv2 = ConvBNRelu(filters, kernel, strides, padding, normalization)
       
    @tf.function 
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        return x
    
class DeResBlock(keras.Model):
    def __init__(self, filters, kernel=3, strides=1, padding='same', use_drop=False, normalization='BatchNormalization'):
        super(DeResBlock, self).__init__()
        self.conv1 = DeConvBNRelu(filters, kernel, strides, padding, use_drop, normalization)
        self.conv2 = DeConvBNRelu(filters, kernel, strides, padding, use_drop, normalization)
       
    @tf.function 
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        return x