import os
import tensorflow as tf
from tensorflow import keras
from config import config

path_weight = os.path.join(config['path_root'], 'models/inceptionV3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

class KID(keras.metrics.Metric):
    def __init__(self, name,  shape=(32,32,3), kid_image_size=75, **kwargs):
        super().__init__(name = name, **kwargs)
        self.shape = shape
        self.kid_tracker = keras.metrics.Mean(name='kid_tracker')
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=shape),
                keras.layers.Rescaling(255.0),
                keras.layers.Resizing(height=kid_image_size, width=kid_image_size),
                #keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
                keras.layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights=None
                    #weights=path_weight
                ),
                keras.layers.GlobalAveragePooling2D()
            ],
            name = 'inception_encoder'
        )
        return
    
    def polynomial_kernel(self, features_1, features_2):
        features_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / features_dimensions + 1.0 ) ** 3.0
    
    def update_state(self, real_images, generated_images, sample_weight=10):
        real_features = self.encoder(real_images, training =False)
        generated_features = self.encoder(generated_images, training = False)
        
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)
        
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / ( batch_size_f * (batch_size_f-1))
        mean_kernel_generated  = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size))) / ( batch_size_f * (batch_size_f-1))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        
        #gen_g_loss_ssim = losses.loss_ssim_batch(real_images, generated_images)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        self.kid_tracker.update_state(kid)
        
    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

        