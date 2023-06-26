from config import config
import os
#from pix2pix import Pix2Pix
import tensorflow as tf
from cycle_pair_gan import CyclePairGan

path_source = os.path.join(config['path_root'], 'data/source/icon-data')
path_source_config = os.path.join(path_source, 'data.json')
cycleGan = CyclePairGan(config['path_root'])

def train_from_start():
    cycleGan.loss_distance = 0.5
    cycleGan.loss_min = 0.5
    cycleGan.init_model()
    cycleGan.optimizer_config = {
            'generator_g_optimizer': [1e-7, 0.01],
            'generator_f_optimizer': [1e-7, 0.01],
            'discriminator_x_optimizer': [1e-6, 0.5],
            'discriminator_y_optimizer': [1e-6, 0.5]
    }
    cycleGan.train(path_source, path_source_config, 50)
    return

def train_from_load():
    cycleGan.loss_distance = 0.5
    cycleGan.loss_min = 0.4
    cycleGan.init_model(True)
    cycleGan.optimizer_config = {
            'generator_g_optimizer': [5e-8, 0.01],
            'generator_f_optimizer': [5e-8, 0.01],
            'discriminator_x_optimizer': [1e-6, 0.5],
            'discriminator_y_optimizer': [1e-6, 0.5]
    }
    cycleGan.train(path_source, path_source_config, 200)
    return

def generate_all():
    cycleGan.init_model(True, False)
    cycleGan.predict_all(path_source, path_source_config)
    return

def generate_all_with_checkpoints(n):
    return

with tf.device('/GPU:0'):
    #train_from_start()
    #train_from_load()
    generate_all()