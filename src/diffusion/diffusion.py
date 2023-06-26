from config import config
import sys
sys.path.insert(0, config['path_public'])
import tensorflow as tf
import os
import shutil
from tensorflow import keras
from kid import KID
from net import get_network, sinusoidal_embedding, ResidualBlock, DownBlock, UpBlock
import losses

class DiffusionModel(keras.Model):
    def __init__(self, image_size, image_channel, widths, block_depth, batch_size):
        super(DiffusionModel,self).__init__()
        self.ema = 0.999
        self.batch_size = batch_size
        self.min_signal_rate = 0.02
        self.max_signal_rate = 0.95
        self.image_channel = image_channel
        self.image_size = image_size
        self.block_depth = block_depth
        self.widths = widths
        self.kid_image_size = int(self.image_size * 1.5)
        self.normalizer = keras.layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)
        
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid", shape=(self.image_size, self.image_size, self.image_channel), kid_image_size=self.kid_image_size)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, self.image_size, self.image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def image_loss_fn(self, images, pred_images):
        return self.loss(images, pred_images) #*0.2 + losses.loss_ssim_batch_shape(images, pred_images)*0.8
    
    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.image_loss_fn(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images, kid_diffusion_steps=5):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.image_loss_fn(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=self.batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}
    
    def config_path(self, save_path):
        self.path_checkpoints = os.path.join(save_path, 'checkpoints/ckpt')
        self.path_progress = os.path.join(save_path, 'progress')
        self.path_generate = os.path.join(save_path, 'generate')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if os.path.exists(self.path_progress):
            shutil.rmtree(self.path_progress)
        if not os.path.exists(self.path_progress):
            os.makedirs(self.path_progress)
        return
    
    def gen_images(self, images_num = 10, plot_diffusion_steps=64, clear=True, black_white = False):
        print(f'start to gen_images | num: {images_num}, steps: {plot_diffusion_steps}')
        if clear and os.path.exists(self.path_generate):
            shutil.rmtree(self.path_generate)
        if not os.path.exists(self.path_generate):
            os.makedirs(self.path_generate)
            
        generated_images = self.generate(
            num_images=images_num,
            diffusion_steps=plot_diffusion_steps
        )
        generated_images = generated_images * 255
        imgs = [generated_images[x] for x in range(generated_images.shape[0])]
        if black_white:
            imgs = tf.image.rgb_to_grayscale(imgs)
        for index, value in enumerate(imgs, start=1):
            tf.keras.preprocessing.image.save_img(f'{self.path_generate}/{plot_diffusion_steps}_{index}.jpg', value)

    def plot_images(self, epoch=None, logs=None, images_num = 3, plot_diffusion_steps=20):
        # plot random generated images for visual evaluation of generation quality
        if epoch % 50 == 49:
            generated_images = self.generate(
                num_images=images_num,
                diffusion_steps=plot_diffusion_steps,
            )

            display_list = tf.concat([generated_images[x] for x in range(generated_images.shape[0])], axis=1)
            display_list = display_list * 255
            #display_list = tf.image.rgb_to_grayscale(display_list)
            tf.keras.preprocessing.image.save_img(f'{self.path_progress}/{epoch}.jpg', display_list)
        # tf.keras.preprocessing.image.save_img(f'{self.path_progress}/_{epoch}.jpg', generated_images[0])
