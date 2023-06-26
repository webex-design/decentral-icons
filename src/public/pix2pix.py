import tensorflow as tf
from layers import ConvBNRelu, DeConvBNRelu, ResBlock, DeResBlock, InstanceNormalization

def generator(input_shape = (32, 32, 1)):
    inputs = tf.keras.layers.Input(input_shape)
    initializer = tf.random_normal_initializer(0., 0.02)
    down_stack = [            
            ConvBNRelu(64, 4),  # (batch_size, 32, 32, 64)
            ConvBNRelu(128, 4),  # (batch_size, 16, 16, 128)
            ConvBNRelu(256, 4),  # (batch_size, 8, 8, 256)
            ConvBNRelu(512, 4),  # (batch_size, 4, 4, 512)
            ConvBNRelu(1024, 4)  # (batch_size, 2, 2, 512)
        ]
    up_stack = [
        DeConvBNRelu(512, 4),  # (batch_size, 4, 4, 1024)
        DeConvBNRelu(256, 4),  # (batch_size, 8, 8, 1024)
        DeConvBNRelu(128, 4),  # (batch_size, 16, 16, 1024)
        DeConvBNRelu(64, 4)  # (batch_size, 32, 32, 512)
    ]
    last = tf.keras.layers.Conv2DTranspose(input_shape[2], 4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh')  # (batch_size, 32, 32, 1)
        
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        #x = tf.keras.Sequential([ResBlock(x.shape[-1]) for i in range(1)])(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip]) #like res-net
        #x = tf.keras.Sequential([DeResBlock(x.shape[-1]) for i in range(1)])(x)

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def discriminator(input_shape = (32, 32, 1), norm_type='instancenorm', use_target=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(input_shape, name='input_image')
    
    if use_target:
        target = tf.keras.layers.Input(input_shape, name='target_image')
        x = tf.keras.layers.concatenate([inputs, target], name='concatenate_input_discriminator_fn')  # (batch_size, 32, 32, channels*2)
    else:
        x = inputs

    down1 = ConvBNRelu(128, 4, normalization = None)(x)  # (batch_size, 16, 16, 64)
    #down1 = tf.keras.Sequential([ResBlock(down1.shape[-1]) for i in range(1)])(down1) 
    down2 = ConvBNRelu(256, 4)(down1)  # (batch_size, 8, 8, 128)
    #down2 = tf.keras.Sequential([ResBlock(down2.shape[-1]) for i in range(1)])(down2)
    down3 = ConvBNRelu(256, 4)(down2)  # (batch_size, 4, 4, 256)
    #down3 = tf.keras.Sequential([ResBlock(down3.shape[-1]) for i in range(1)])(down3)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 6, 6, 256)
    conv = tf.keras.layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer,use_bias=False)(zero_pad1)  # (batch_size, 4, 4, 512)
    
    batchnorm1 = InstanceNormalization()(conv) if norm_type.lower() == 'instancenorm' else tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 4, 4, 512)
    last = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer, name='last_layer_conv2d')(zero_pad2)  # (batch_size, 30, 30, 1)
    
    if use_target:
        return tf.keras.Model(inputs=[inputs, target], outputs=last)
    else:
        return tf.keras.Model(inputs=inputs, outputs=last)
    