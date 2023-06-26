import os
import math
import numpy as np
import tensorflow as tf
import datetime
import shutil
import time
from unit import ConvBNRelu, DeConvBNRelu, ResBlock, DeResBlock
from my_data_pair import MyData
from tensorflow import keras
from tensorflow.image import ssim

class Pix2Pix:
    def __init__(self, root, channels=1, name = 'pix2pix'):
        self.root = root
        self.channels = channels
        root_model = os.path.join(root, f'models/{name}')
        self.paths = {}
        self.paths['root'] = root
        self.paths['root_model'] = os.path.join(root, f'models/{name}')
        self.paths['training_checkpoints_prefix'] = os.path.join(root_model, 'training_checkpoints', 'ckpt')
        self.paths['training_checkpoints'] = os.path.join(root_model, 'training_checkpoints')
        self.paths['save_g'] = os.path.join(root_model, 'pix2pix_g.h5')
        self.paths['save_d'] = os.path.join(root_model, 'pix2pix_d.h5')
        self.paths['log_images'] = os.path.join(root_model, 'log_images')
        self.paths['log_fit'] = os.path.join(root_model, 'log_fit')
        self.lambdas = {}
        self.lambdas['gan'] = 0.7
        self.lambdas['ssim_input'] = 0
        self.lambdas['ssim_target'] = 0.5
        self.lambdas['l1'] = 0.2
        self.distance = 0.2
        self.my_data = MyData(self.channels)
        self.summary_writer = tf.summary.create_file_writer(self.paths['log_fit'] + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        return
    
    def init_model(self, generator_path=None, discriminator_path=None, needClear=True):
        with tf.keras.utils.custom_object_scope({'ConvBNRelu': ConvBNRelu , 'DeConvBNRelu': DeConvBNRelu, 'ResBlock': ResBlock, 'DeResBlock': DeResBlock}):
        # init generator
            if generator_path is None:
                self.__generator_fn()
            else:
                self.generator = tf.keras.models.load_model(generator_path)

            # init discriminator
            if discriminator_path is None:
                self.__discriminator_fn()
            else:
                self.discriminator = tf.keras.models.load_model(discriminator_path)
            
        if(needClear):
            self.clear()
            
        self.generator_optimizer = tf.keras.optimizers.legacy.Adam(5e-5, beta_1=0.01) # init 5e-5, beta_1=0.01
        self.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(5e-5, beta_1=0.5) # init 5e-5, beta_1=0.5
        self.__checkpoint_fn()
        
    # after optimizer
    def __checkpoint_fn(self):
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        self.checkpoint = checkpoint
        
    def __checkDictectory(self, dir, removeOld = False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif removeOld:
            print('remove path',dir)
            shutil.rmtree(dir)
        return
      
    def clear(self):
        self.__checkDictectory(self.paths['root_model'], True)
        os.makedirs(self.paths['log_images'])
        os.makedirs(self.paths['log_fit'])
        os.makedirs(self.paths['training_checkpoints'])
    
    def __generator_fn(self):
        inputs = tf.keras.layers.Input(shape=(32, 32, self.channels))
        initializer = tf.random_normal_initializer(0., 0.02)
        down_stack = [            
            ConvBNRelu(64, 4),  # (batch_size, 32, 32, 64)
            ConvBNRelu(128, 4),  # (batch_size, 16, 16, 128)
            ConvBNRelu(256, 4),  # (batch_size, 8, 8, 256)
            ConvBNRelu(512, 4),  # (batch_size, 4, 4, 512)
            ConvBNRelu(1024, 4)  # (batch_size, 2, 2, 512)
        ]
        up_stack = [
            DeConvBNRelu(512, 4, use_drop=False),  # (batch_size, 4, 4, 1024)
            DeConvBNRelu(256, 4, use_drop=False),  # (batch_size, 8, 8, 1024)
            DeConvBNRelu(128, 4),  # (batch_size, 16, 16, 1024)
            DeConvBNRelu(64, 4)  # (batch_size, 32, 32, 512)
        ]
    
        # make sure the value in [-1,1]
        last = tf.keras.layers.Conv2DTranspose(self.channels, 4,
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
        self.generator = tf.keras.Model(inputs=inputs, outputs=x)
        return self.generator
        
    def __discriminator_fn(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inputs = tf.keras.layers.Input(shape=(32, 32, self.channels), name='input_image')
        target = tf.keras.layers.Input(shape=(32, 32, self.channels), name='target_image')

        x = tf.keras.layers.concatenate([inputs, target], name='concatenate_input_discriminator_fn')  # (batch_size, 32, 32, channels*2)

        down1 = ConvBNRelu(128, 4, use_bn = False)(x)  # (batch_size, 16, 16, 64)
        down1 = tf.keras.Sequential([ResBlock(down1.shape[-1]) for i in range(1)])(down1)
        down2 = ConvBNRelu(256, 4)(down1)  # (batch_size, 8, 8, 128)
        down2 = tf.keras.Sequential([ResBlock(down2.shape[-1]) for i in range(1)])(down2)
        down3 = ConvBNRelu(256, 4)(down2)  # (batch_size, 4, 4, 256)
        down3 = tf.keras.Sequential([ResBlock(down3.shape[-1]) for i in range(1)])(down3)
        
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 6, 6, 256)
        conv = tf.keras.layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer,use_bias=False)(zero_pad1)  # (batch_size, 4, 4, 512)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 4, 4, 512)
        last = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer, name='last_layer_conv2d')(zero_pad2)  # (batch_size, 30, 30, 1)
        self.discriminator = tf.keras.Model(inputs=[inputs, target], outputs=last)
        return self.discriminator

    def generator_loss(self, disc_generated_output, gen_output, target, input_image):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        ssim_input_loss = 1 - tf.reduce_mean(ssim(input_image, gen_output, max_val=1.0))
        ssim_target_loss = 1 - tf.reduce_mean(ssim(target, gen_output, max_val=1.0))
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = self.lambdas['gan'] * gan_loss + self.lambdas['l1'] * l1_loss + self.lambdas['ssim_input'] * ssim_input_loss + self.lambdas['ssim_target'] * ssim_target_loss 
        return total_gen_loss, gan_loss, l1_loss, ssim_input_loss, ssim_target_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
          
    # Normalizing the images [-1,1] to [0, 255]   
    def generate_images(self, test_input, tar, step):
        prediction = self.generator(test_input, training=False)
        display_list = tf.concat([test_input[0], tar[0], prediction[0]], axis=1)
        display_list = (display_list + 1 ) * 127.5
        tf.keras.preprocessing.image.save_img(os.path.join(self.paths['log_images'], f'predict_{step}.jpg'), display_list)
        
    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image[0], target[0]], training=True)
            disc_generated_output = self.discriminator([input_image[0], gen_output[0]], training=True)
            gen_total_loss, gen_gan_loss, gen_l1_loss, ssim_loss, ssim_loss2 = self.generator_loss(disc_generated_output, gen_output, target, input_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        
        if(disc_loss > 0.5 or disc_loss - gen_total_loss > self.distance):
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        if(gen_total_loss > 0.5 or gen_total_loss - disc_loss >= self.distance):
            generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            
        #with self.summary_writer.as_default():
            #tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            #tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            #tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            #tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, ssim_loss, ssim_loss2
          
    def train(self,source_path, source_config_path, steps=100000):
        self.my_data.read(source_path, source_config_path)
        dataset_train = self.my_data.create_dataset()
        self.generator.compile()
        self.discriminator.compile()
        self.__fit(dataset_train, steps=steps)
        self.__save()
        return
    
    def __fit(self, train_dataset, steps):
        example_input, example_target, example_labels = next(iter(train_dataset.take(1)))
        start = time.time()
        print('DEBUG_____', int(steps))
        for step, (input_image, target, labels) in train_dataset.repeat().take(steps).enumerate():
            if (step) % 500 == 0:
                if step != 0:
                    print(f'Step: {step//1} | Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
                    
                start = time.time()
                # generate
                self.generate_images(example_input, example_target, step)
            
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, ssim_loss, ssim_loss2 = self.train_step(input_image, target, step)

             # Training step
            if (step+1) % 50 == 0:
                print(f'loss==> disc - {disc_loss} | total - {gen_total_loss} | gan: {gen_gan_loss} | l1: {gen_l1_loss} |  ssim_target: {ssim_loss} |  ssim_input: {ssim_loss2}')

            # Save (checkpoint) the model every 2k steps
            if (step + 1) % 2000 == 0:
                self.checkpoint.save(file_prefix=self.paths['training_checkpoints_prefix'])
        
    def __save(self):
        self.generator.save(self.paths['save_g'])
        self.discriminator.save(self.paths['save_d'])
        return
    
    def generate_all_fn(self, ck_total, source_path, source_config_path, out_path, check_prex = None):
        self.my_data.read_source(source_path, source_config_path)
        dataset_train = self.my_data.create_dataset()
        total = self.my_data.num_label
        side_length = int(math.sqrt(total))
        rows = math.ceil( total / side_length)
        toadd = side_length * rows - total 
        
        prefix =  self.paths['training_checkpoints_prefix'] if check_prex is None else check_prex
        
        for i in reversed(range(ck_total)):
            self.checkpoint.restore(f'{prefix}-{i+1}').expect_partial()
            self.generate_all(out_path, str(i+1), dataset_train, total ,rows, toadd)

        # self.generate_all(out_path, str(i+1), dataset_train, total ,rows, toadd)
    
    def generate_all(self, out_path, name, dataset_train, total, rows, toadd):
        list = []
        list_ipt = []
        for step, (input_image, target, labels) in dataset_train.take(total).enumerate():
            prediction = self.generator(input_image, training=False)
            list.append(prediction[0])
            list_ipt.append(input_image[0])
            
        prediction_outs = self.combine_predict(list, rows, toadd)
        ipt_outs = self.combine_predict(list_ipt, rows, toadd)
    
        tf.keras.preprocessing.image.save_img(os.path.join(out_path, f'all-{name}.jpg'), tf.concat([prediction_outs, ipt_outs], axis=0))
        print(f'Saved checkpoint images - {name}!')
        return
    
    def combine_predict(self, list, rows, toadd):
        for i in range(toadd):
            list.append(tf.fill((32, 32, self.channels), 1))
        
        lists = np.array_split(list, rows)
        
        tflist = []
        for element in lists:
            newlist = tf.concat(element, axis=0)
            newlist = tf.reshape(newlist, (32 * len(element), 32, 1))
            tflist.append(newlist)
            
        concatenated_tensor = tf.concat(tflist, axis=1)
        concatenated_tensor = (concatenated_tensor + 1 ) * 127.5
        concatenated_tensor = tf.where(concatenated_tensor < 100, 0, concatenated_tensor)
        return concatenated_tensor