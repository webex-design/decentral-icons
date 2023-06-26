import tensorflow as tf
from tensorflow import keras

class ConvBNRelu(keras.Model):
    def __init__(self, filters, kernel=3, strides=1, padding='same', use_bn=True):
        super(ConvBNRelu, self).__init__()
        self.use_bn = use_bn
        self.conv = keras.layers.Conv2D(filters, kernel, strides, padding)
        self.relu = keras.layers.ReLU()
        self.bn = None
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization()
    
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
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        return x
    
class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv0 = ConvBNRelu(64)
        self.conv1 = ConvBNRelu(128)
        self.conv2 = ConvBNRelu(256)
        self.conv3 = ConvBNRelu(512)
        self.conv4 = ConvBNRelu(512)
        self.conv5 = ConvBNRelu(512, False)
        
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
    
class WNet(keras.Model):
    def __init__(self, m):
        super(Encoder, self).__init__()
        self.left = Encoder()
        self.right = Encoder()
        self.left_1 = tf.keras.Sequential([ResBlock(32) for i in range(m-4)])
        self.left_2 = tf.keras.Sequential([ResBlock(128) for i in range(m-2)])
        self.left_3 = tf.keras.Sequential([ResBlock(256) for i in range(m)])
        self.right_3 = tf.keras.Sequential([ResBlock(256) for i in range(m)])
        
        self.deconv1 = DeConvBNRelu(512, use_bn=False)
        self.deconv2 = DeConvBNRelu(512)
        self.deconv3 = DeConvBNRelu(256)
        self.deconv4 = DeConvBNRelu(128)
        self.deconv5 = DeConvBNRelu(64)
        self.deconv6 = DeConvBNRelu(1)
        
    def call(self, inputs_l, inputs_r):
        l_out = self.left(inputs_l)
        r_out = self.right(inputs_r)
        
        lout_0 = self.left_1(l_out.conv0)
        lout_1 = self.left_2(l_out.conv1)
        lout_2 = self.left_3(l_out.conv2)
        lout_3 = l_out.conv3
        lout_4 = l_out.conv4
        lout_5 = l_out.conv5
        rout_2 = self.right_3(r_out.conv2)
        rout_3 = r_out.conv3
        rout_4 = r_out.conv4
        rout_5 = r_out.conv5
        # for c in [lout_5, rout_5]:
        #     print("cshape", c.shape)
        de_0 = self.deconv1(tf.concat([lout_5, rout_5], axis=1))
        de_1 = self.deconv2(tf.concat([lout_4, de_0, rout_4], axis=1))
        de_2 = self.deconv3(tf.concat([lout_3, de_1, rout_3], axis=1))
        de_3 = self.deconv4(tf.concat([lout_2, de_2, rout_2], axis=1))
        de_4 = self.deconv5(tf.concat([lout_1, de_3], axis=1))
        de_5 = self.deconv6(tf.concat([lout_0, de_4], axis=1))
        return de_5, lout_5, rout_5
    
class Discriminator(keras.Model):
    def __init__(self, num_fonts=80, num_characters=3500 + 1):
        super(Encoder, self).__init__()
        self.conv0 = ConvBNRelu(64, kernel=5, strides=2, padding='same', use_bn=False)
        self.conv1 = ConvBNRelu(128, kernel=5, strides=2)
        self.conv2 = ConvBNRelu(256, kernel=5, strides=2)
        self.conv3 = ConvBNRelu(512, kernel=5, strides=1)
        self.dense0 = keras.layers.Dense(units=1, activation='sigmoid')
        self.dense1 = keras.layers.Dense(units=num_fonts, activation='softmax')
        self.dense2 = keras.layers.Dense(units=num_characters, activation='softmax')
        
    def call(self, x1, x2, x3):
        x = tf.concat([x1, x2, x3], axis=1)
        features = []
        x = self.conv0(x)
        features.append(x)
        x = self.conv1(x)
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        x = self.flatten(x)
        x1 = self.dense0(x)  # real or fake
        x2 = self.dense1(x)  # font category
        x3 = self.dense2(x)  # char category
        return x1, x2, x3, features

class ClSEncoderP(tf.keras.Model):
    def __init__(self, num_characters=3375):
        super(ClSEncoderP, self).__init__()
        self.fc = tf.keras.layers.Dense(num_characters)

    def call(self, x):
        return self.fc(x)


class CLSEncoderS(tf.keras.Model):
    def __init__(self, num_fonts=80):
        super(CLSEncoderS, self).__init__()
        self.fc = tf.keras.layers.Dense(num_fonts)

    def call(self, x):
        return self.fc(x)
    