import os
from pix2pix import Pix2Pix
import tensorflow as tf

current_file_dir = os.path.dirname(os.path.realpath(__file__))
path_root = os.path.abspath(os.path.join(current_file_dir, '../../'))
path_source = os.path.join(path_root, 'data/source/icon-data')
path_source_config = os.path.join(path_source, 'data.json')
path_predict = os.path.join(path_root, 'predict')

g_path = os.path.join(path_root, 'models/pix2pix_g.h5')
d_path = os.path.join(path_root, 'models/pix2pix_d.h5')
checkpoints_pre =os.path.join(path_root, 'models/pix2pix/training_checkpoints/ckpt')

print(checkpoints_pre)

pix2pix_model = Pix2Pix(path_root)

def train_from_start():
    pix2pix_model.init_model()
    pix2pix_model.generator_optimizer.learning_rate = 1e-6
    pix2pix_model.generator_optimizer.beta_1 = 0.01
    pix2pix_model.discriminator_optimizer.learning_rate = 1e-6
    pix2pix_model.discriminator_optimizer.beta_1 = 0.5
    pix2pix_model.lambdas['gan'] = 0.5
    pix2pix_model.lambdas['ssim_input'] = 0
    pix2pix_model.lambdas['ssim_target'] = 0.5
    pix2pix_model.lambdas['l1'] = 0
    pix2pix_model.generator.summary()
    pix2pix_model.train(path_source, path_source_config, 200000)
    return

def train_from_load():
    pix2pix_model.init_model(g_path, d_path)
    pix2pix_model.generator_optimizer.learning_rate = 1e-10
    pix2pix_model.generator_optimizer.beta_1 = 0.05
    pix2pix_model.discriminator_optimizer.learning_rate = 1e-6
    pix2pix_model.discriminator_optimizer.beta_1 = 0.5
    pix2pix_model.lambdas['gan'] = 0.4
    pix2pix_model.lambdas['ssim_input'] = 0
    pix2pix_model.lambdas['ssim_target'] = 0.6
    pix2pix_model.lambdas['l1'] = 0
    pix2pix_model.distance = 0.4
    pix2pix_model.generator.summary()
    pix2pix_model.train(path_source, path_source_config, 100000)
    return

def generate_all():
    return

def generate_all_with_checkpoints(n):
    pix2pix_model.init_model(g_path, d_path, False)
    pix2pix_model.generate_all_fn(n, path_source, path_source_config, path_predict, checkpoints_pre)
    return

with tf.device('/GPU:0'):
    #train_from_start()
    #train_from_load()
    generate_all_with_checkpoints(1)
    