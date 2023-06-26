import tensorflow as tf
from tensorflow import keras

def MY_LOSS(y_true, y_pred):
    # Compute SSIM
    #ssim = 1 - tf.image.ssim(y_true, y_pred, max_val=255.0, k1=0.001)
    mae = keras.backend.mean(keras.backend.abs(y_true - y_pred))
    return mae

def PSNR(super_resolution, high_resolution):
        """Compute the peak signal-to-noise ratio, measures quality of image."""
        # Max value of pixel is 255
        psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
        return psnr_value

class ResBlock(keras.Model):
    def __init__(self, filter):
        super(ResBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(filter, 3, padding="same", activation="relu")
        self.conv2 = keras.layers.Conv2D(filter, 3, padding="same")
       
    @tf.function 
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = keras.layers.Add()([inputs, x])
        return x
    
class Upsampling(keras.Model):
    def __init__(self, filter, factor=2, **kwargs):
        super(Upsampling, self).__init__()
        self.factor = factor
        self.conv1 = keras.layers.Conv2D(filter * (factor ** 2), 3, padding="same", **kwargs)
        self.conv2 = keras.layers.Conv2D(filter * (factor ** 2), 3, padding="same", **kwargs)
        
    @tf.function 
    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.nn.depth_to_space(x, block_size=self.factor)
        x = self.conv2(x)
        x = tf.nn.depth_to_space(x, block_size=self.factor)
        return x
    
class EDSRModel(keras.Model):
    def __init__(self, num_filters, num_of_residual_blocks):
        super(EDSRModel, self).__init__()
        self.num_filters = num_filters
        self.num_of_residual_blocks = num_of_residual_blocks
    
    def get_config(self):
        config = super(EDSRModel, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'num_of_residual_blocks': self.num_of_residual_blocks
        })
        return config 

    def build(self, input_shape):
        # Define the layers in the model
        self.input_layer = keras.layers.Input(shape=(None, None, None, 3))
        self.rescaling = keras.layers.Rescaling(scale=1.0 / 255)
        self.conv1 = keras.layers.Conv2D(self.num_filters, 3, padding="same")
        self.res = tf.keras.Sequential([ResBlock(self.num_filters) for _ in range(self.num_of_residual_blocks)])
        self.conv2 = keras.layers.Conv2D(self.num_filters, 3, padding="same")
        self.upsample = Upsampling(self.num_filters)
        self.conv3 = tf.keras.layers.Conv2D(3, 3, padding="same")
        self.rescaling2 = tf.keras.layers.Rescaling(scale=255)

    @tf.function
    def call(self, inputs, training):
        x = self.rescaling(inputs)
        x_new = x = self.conv1(x)
        x_new = self.res(x_new)
        x_new = self.conv2(x_new)
        x = keras.layers.Add()([x, x_new])
        x = self.upsample(x)
        x = self.conv3(x)
        output = self.rescaling2(x)
        return output
    
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, x):
        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        # Passing low resolution image to model
        super_resolution_img = self.call(x, training=False)
        # Clips the tensor from min(0) to max(255)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        #super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 1)
        # Rounds the values of a tensor to the nearest integer
        super_resolution_img = tf.round(super_resolution_img)
        # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
        super_resolution_img = tf.squeeze(
            tf.cast(super_resolution_img, tf.uint8), axis=0
        )
        return super_resolution_img
