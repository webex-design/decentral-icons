import os
import tensorflow as tf
from diffusion import DiffusionModel
from tensorflow import keras
from dataset import Datas
from config import config
import tensorflow_addons as tfa

USE_CK=True
path_source = os.path.join(config['path_root'], 'data/style/shanshui')
path_source2 = os.path.join(config['path_root'], 'data/output/icon-data')

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
        super(LearningRateScheduler, self).__init__()
        self.optimizer = optimizer
        self.range = 1

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        if epoch == 500:
            tf.keras.backend.set_value(self.optimizer.learning_rate, 1e-8)
        #current_lr = tf.keras.backend.get_value(self.optimizer.learning_rate)
        #loss = logs['i_loss']
        #self.switch_lt(1, loss, 0.2, 1000000, 1e-7)
        #self.switch_lt(3, loss, 0.15, 0.2, 1e-7)
        #self.switch_lt(3, loss, 0.13, 0.15, 1e-7)
        #self.switch_lt(4, loss, 0.11, 0.13, 1e-7)
        #self.switch_lt(5, loss, 0.0, 0.11, 1e-8)
            
    def switch_lt(self, range, loss,loss_min, loss_max, learning_rate):
        if loss>loss_min and loss<=loss_max and self.range!=range:
            print(f'change optimizer -> {range}')
            self.range=range
            tf.keras.backend.set_value(self.optimizer.learning_rate, learning_rate)

def restore(model):
    model.load_weights(model.path_checkpoints)
    print('load checkpoints')

def run():
    epochs = 200
    batch_size = 40
    image_size = 128
    image_channel = 3
    widths = [32, 64, 96, 128, 256]
    block_depth = 2 
    learning_rate = 1e-4 #1e-4
    weight_decay = 1e-3
    beta_1 = 0.9
    #num_epochs = 50  # train for at least 50 epochs for good results

    mydatas = Datas(image_shape = (image_size, image_size, image_channel), batch_size = batch_size)
    mydatas.read_folder(path_source, 1)
    # mydatas.read_max(path_source2, 1)
    train_dataset = mydatas.create_dataset_train()
    val_dataset = mydatas.create_dataset_valid()

    # create and compile the model
    model = DiffusionModel(image_size,image_channel, widths, block_depth, batch_size)
    model.config_path(os.path.join(config['path_root'], 'models/diffusion'))
    # below tensorflow 2.9:
    # pip install tensorflow_addons
    # import tensorflow_addons as tfa
    # optimizer=tfa.optimizers.AdamW

    
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_1 = beta_1),
        #optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, beta_1 = beta_1),
        loss=keras.losses.mean_absolute_error,
    )
    
    if(USE_CK):
        restore(model)
    # pixelwise mean absolute error is used as loss
    
    lr_scheduler = LearningRateScheduler(model.optimizer)
    # save the best model based on the validation KID metric
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model.path_checkpoints,
        save_weights_only=True,
        monitor="val_kid",
        mode="min",
        save_best_only=True,
        save_freq = 'epoch'
    )

    # calculate mean and variance of training dataset for normalization
    # model.normalizer.adapt(train_dataset)
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
            #lr_scheduler,
            checkpoint_callback,
        ],
    )
    model.gen_images(10, 400)
    model.gen_images(20, 200, False, True)
    #model.gen_images(20, 50)
    #model.gen_images(10, 20)
    # run training and plot generated images periodically
    #model.save(os.path.join(model.path_model, 'save_g1'))
    
with tf.device('/GPU:0'):
    run()
