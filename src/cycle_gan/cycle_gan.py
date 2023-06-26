from config import config
import sys
sys.path.insert(0, config['path_public'])
import os
import numpy as np
import tensorflow as tf
import pix2pix
from  datas import Datas
import time
import shutil
from layers import ConvBNRelu, DeConvBNRelu, ResBlock, DeResBlock, InstanceNormalization
from predict import generate_all

class CycleGan:
    def __init__(self, root , input_shape = (32, 32, 1)):
        self.loss_distance = 0.5
        self.loss_min = 0.2
        self.input_shape = input_shape
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.LAMBDA = 10
        self.datas = Datas(input_shape)
        name = 'CycleGan'
        root_model = os.path.join(root, f'models/{name}')
        self.paths = {}
        self.paths['root'] = root
        self.paths['root_model'] = os.path.join(root, f'models/{name}')
        self.paths['training_checkpoints_prefix'] = os.path.join(root_model, 'training_checkpoints', 'ckpt')
        self.paths['training_checkpoints'] = os.path.join(root_model, 'training_checkpoints')
        self.paths['models'] = os.path.join(root_model, 'models')
        self.paths['save_g1'] = os.path.join(self.paths['models'], 'cycle_g1.h5')
        self.paths['save_g2'] = os.path.join(self.paths['models'], 'cycle_g2.h5')
        self.paths['save_d1'] = os.path.join(self.paths['models'], 'cycle_d1.h5')
        self.paths['save_d2'] = os.path.join(self.paths['models'], 'cycle_d2.h5')
        self.paths['log_images'] = os.path.join(root_model, 'log_images')
        self.paths['log_fit'] = os.path.join(root_model, 'log_fit')
        self.paths['predicts'] = os.path.join(root_model, 'predicts')
        self.optimizer_config = {
            'generator_g_optimizer': [1e-5, 0.01],
            'generator_f_optimizer': [1e-5, 0.01],
            'discriminator_x_optimizer': [1e-5, 0.5],
            'discriminator_y_optimizer': [1e-5, 0.5]
        }
        return
    
    def __checkpoint_fn(self):
        self.checkpoint = tf.train.Checkpoint(generator_g=self.generator_g,
                            generator_f=self.generator_f,
                            discriminator_x=self.discriminator_x,
                            discriminator_y=self.discriminator_y,
                            generator_g_optimizer=self.generator_g_optimizer,
                            generator_f_optimizer=self.generator_f_optimizer,
                            discriminator_x_optimizer=self.discriminator_x_optimizer,
                            discriminator_y_optimizer=self.discriminator_y_optimizer)

    # Normalizing the images [-1,1] to [0, 255]   
    def __generate_images(self, test_input, test_tar, epoch, step):
        prediction_g = self.generator_g(test_input, training=True)
        prediction_g2 = self.generator_f(prediction_g, training=True)
        prediction_f = self.generator_f(test_tar, training=True)
        prediction_f2 = self.generator_g(prediction_f, training=True)
        display_list = tf.concat([test_input[0], prediction_g[0], prediction_g2[0], test_tar[0], prediction_f[0], prediction_f2[0]], axis=1)
        display_list = (display_list + 1 ) * 127.5
        tf.keras.preprocessing.image.save_img(os.path.join(self.paths['log_images'], f'predict_{epoch}_{step}.jpg'), display_list)
    
    def __init_dir(self, dir, removeOld = False):
        if removeOld and os.path.exists(dir):
            shutil.rmtree(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return
        
    def clear(self):
        self.__init_dir(self.paths['root_model'])
        self.__init_dir(self.paths['log_images'], True)
        self.__init_dir(self.paths['log_fit'], True)
        self.__init_dir(self.paths['training_checkpoints'], True)
        self.__init_dir(self.paths['predicts'], True)
        self.__init_dir(self.paths['models'])
    
    def init_model(self, models=False, needClear=True):
        with tf.keras.utils.custom_object_scope({
            'InstanceNormalization': InstanceNormalization,
            'ConvBNRelu': ConvBNRelu, 
            'DeConvBNRelu': DeConvBNRelu,
            'ResBlock': ResBlock, 
            'DeResBlock': DeResBlock,
            'pix2pix': pix2pix
        }):
            
            if models:
                self.generator_g = tf.keras.models.load_model(self.paths['save_g1'])
                self.generator_f = tf.keras.models.load_model(self.paths['save_g2'])
                self.discriminator_x = tf.keras.models.load_model(self.paths['save_d1'])
                self.discriminator_y = tf.keras.models.load_model(self.paths['save_d2'])
            else:
                self.generator_g = pix2pix.generator(self.input_shape)
                self.generator_f = pix2pix.generator(self.input_shape)
                self.discriminator_x = pix2pix.discriminator(self.input_shape)
                self.discriminator_y = pix2pix.discriminator(self.input_shape)
            
        if(needClear):
            self.clear()
            
        self.set_optimizers()
        self.__checkpoint_fn()
        return
    
    def set_optimizers(self):
        __c = self.optimizer_config
        self.generator_g_optimizer = tf.keras.optimizers.legacy.Adam(__c['generator_g_optimizer'][0], beta_1=__c['generator_g_optimizer'][1])
        self.generator_f_optimizer = tf.keras.optimizers.legacy.Adam(__c['generator_f_optimizer'][0], beta_1=__c['generator_f_optimizer'][1])
        self.discriminator_x_optimizer = tf.keras.optimizers.legacy.Adam(__c['discriminator_x_optimizer'][0], beta_1=__c['discriminator_x_optimizer'][1])
        self.discriminator_y_optimizer = tf.keras.optimizers.legacy.Adam(__c['discriminator_y_optimizer'][0], beta_1=__c['discriminator_y_optimizer'][1])

    def generator_loss(self, generated):
        return self.loss_object(tf.ones_like(generated), generated)
        
    def discriminator_loss(self, real, generated):
        real_loss = self.loss_object(tf.ones_like(real), real)
        generated_loss = self.loss_object(tf.zeros_like(generated), generated)
        return (real_loss + generated_loss) * 0.5

    def calc_cycle_loss(self,real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss
    
    @tf.function
    def train_step(self, real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

            # Calculate the gradients for generator and discriminator
            generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
            generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
            discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
            discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

            # Apply the gradients to the optimizer
            if(gen_g_loss > self.loss_min or gen_g_loss - disc_y_loss > self.loss_distance):
                self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g.trainable_variables))
            if(gen_f_loss > self.loss_min or gen_f_loss - disc_x_loss > self.loss_distance):
                self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f.trainable_variables))
            if(disc_x_loss > self.loss_min or disc_x_loss - gen_f_loss > self.loss_distance):
                self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, self.discriminator_x.trainable_variables))
            if(disc_y_loss > self.loss_min or disc_y_loss - gen_g_loss > self.loss_distance):
                self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, self.discriminator_y.trainable_variables))
            return total_cycle_loss, total_gen_g_loss, total_gen_f_loss, gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss
 
    def train(self, source_path, source_config_path, steps=50):
        self.generator_g.compile()
        self.generator_f.compile()
        self.discriminator_x.compile()        
        self.discriminator_y.compile()
        self.datas.read(source_path, source_config_path)
        source_images, handdrawn_images, labels = self.datas.source
        dataset_source = self.datas.create_dataset_one(source_images)
        dataset_target = self.datas.create_dataset_one(handdrawn_images)
        example_input = next(iter(dataset_source.take(1)))
        example_target = next(iter(dataset_target.take(1)))
        for epoch in range(steps):
            start = time.time()
            print (f'start to epoch {epoch} / {steps} times.')
            n = 0
            for image_x, image_y in tf.data.Dataset.zip((dataset_source, dataset_target)):
                total_cycle_loss, total_gen_g_loss, total_gen_f_loss, gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = self.train_step(image_x, image_y)
                if n % 20 == 0:
                    print(f'{n} => cycle {total_cycle_loss} | total_gen {total_gen_g_loss} / {total_gen_f_loss} | gen: {gen_g_loss} / {gen_f_loss} | disc: {disc_x_loss} / {disc_y_loss} ')
                #if n % 100 == 0:
                    #self.__generate_images(example_input, example_target, epoch, n)
                n+=1
            # 使用一致的图像（sample_horse），以便模型的进度清晰可见。
            self.__generate_images(example_input, example_target, epoch, n)

            #if (epoch + 1) % 5 == 0:
                #self.checkpoint.save(file_prefix=self.paths['training_checkpoints_prefix'])

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

        self.save()
        return
    
    def save(self):
        self.generator_g.save(self.paths['save_g1'])
        self.generator_f.save(self.paths['save_g2'])
        self.discriminator_x.save(self.paths['save_d1'])        
        self.discriminator_y.save(self.paths['save_d2'])
        return

    def predict_all(self, source_path, source_config_path): 
        datas = Datas(self.input_shape)    
        datas.read_all(source_path, source_config_path)
        source_images, handdrawn_images, labels = datas.source

        dataset_source = datas.create_dataset_one(source_images)
        dataset_target = datas.create_dataset_one(handdrawn_images)
        
        generate_all(self.generator_g, self.paths['predicts'], 'g0', dataset_source, len(source_images))
        generate_all(self.generator_f, self.paths['predicts'], 'f0', dataset_target, len(handdrawn_images))
        return        
        