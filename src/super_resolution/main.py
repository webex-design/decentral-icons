import tensorflow as tf
from model import ResBlock, Upsampling, EDSRModel, PSNR, MY_LOSS
from tensorflow import keras
from super_dataset import Super_Dataset
import os
from config import config
import shutil

class Main():
    def __init__(self):
        self.high_paths = os.path.join(config['path_root'], 'data/DIV2K/DIV2K_valid_HR')
        self.low_paths = os.path.join(config['path_root'], 'data/DIV2K/DIV2K_valid_LR')
        self.model_path = os.path.join(config['path_root'], 'models/super/models/m')
        self.predict_path = os.path.join(config['path_root'], 'models/super/predict')
        self.path_style = os.path.join(config['path_root'], 'data/style')
        self.predict_path_img = os.path.join(config['path_root'], 'data/DIV2K/DIV2K_valid_LR/0808x4.png')
        self.predict_path_img2 = os.path.join(config['path_root'], 'data/style/culture/paper_cut.jpg')
        self.path_checkpoints = os.path.join(config['path_root'], 'models/super/checkpoints/ckpt')
        if os.path.exists(self.predict_path):
            shutil.rmtree(self.predict_path)
        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)

        self.sds = Super_Dataset(0.8)   
        self.LOW_IMAGE = self.sds.load(self.predict_path_img)
        self.LOW_IMAGE2 = self.sds.load(self.predict_path_img2)
            
    def load(self):
        with tf.keras.utils.custom_object_scope({
            'ResBlock': ResBlock, 
            'Upsampling': Upsampling,
            'PSNR': PSNR
        }):
            self.model = tf.keras.models.load_model(self.model_path, custom_objects={'EDSRModel': EDSRModel})
            self.complie()

    def load_weight(self):
        self.create()
        self.model.load_weights(self.path_checkpoints)
        
    def complie(self):
        self.model.compile(
            optimizer=keras.optimizers.legacy.Adam(
                learning_rate= 1e-6,
                beta_1 = 0.9,
                epsilon = 1e-10
            ), 
            loss=MY_LOSS,
            metrics=[PSNR]
        )
            
    def create(self):
        self.model = EDSRModel(num_filters=128, num_of_residual_blocks=20)
        self.complie()
        #self.model = EDSRModel(num_filters=64, num_of_residual_blocks=16)
        
    def predict(self, input, output):
        display_list = self.model.predict(input)
        tf.keras.preprocessing.image.save_img(output, display_list)
        return
        
    def train_call_back(self, epoch=None, logs=None):
        if epoch % 20 == 19:
            self.predict(self.LOW_IMAGE, f'{self.predict_path}/training_{epoch}.jpg')
        return

    def train(self, epochs=1):
        self.sds.read(self.low_paths, self.high_paths)
        ds, ds2 = self.sds.create_dataset()
        print('Start to fit...')
        self.model.fit(
            ds, 
            epochs=epochs,
            validation_data = ds2,
            callbacks=[
                keras.callbacks.LambdaCallback(on_epoch_end=self.train_call_back)
            ])
        
    def p(self, path_ipt, path_outer):
        self.predict(
            self.sds.load(os.path.join(self.path_style, path_ipt)),
            path_outer
        )

    def gen(self):
        self.predict(self.LOW_IMAGE2, f'{self.predict_path}/_.jpg')
        self.p('flower/1.jpg', f'{self.predict_path}/_1.jpg')
        self.p('flower/2.jpg', f'{self.predict_path}/_2.jpg')
        self.p('flower/3.jpg', f'{self.predict_path}/_3.jpg')
        self.p('flower/4.jpg', f'{self.predict_path}/_4.jpg')
        self.p('flower/5.jpg', f'{self.predict_path}/_5.jpg')
        self.p('dragon/1.jpg', f'{self.predict_path}/_6.jpg')
        self.p('dragon/2.jpg', f'{self.predict_path}/_7.jpg')
        self.p('dragon/3.jpg', f'{self.predict_path}/_8.jpg')
        self.p('dragon/4.jpg', f'{self.predict_path}/_9.jpg')
        self.p('dragon/5.jpg', f'{self.predict_path}/_10.jpg')
        self.p('shanshui/9.jpg', f'{self.predict_path}/_11.jpg')
        self.p('shanshui/8.jpg', f'{self.predict_path}/_12.jpg')
        self.p('shanshui/7.jpg', f'{self.predict_path}/_13.jpg')
        self.p('shanshui/6.jpg', f'{self.predict_path}/_14.jpg')
        self.p('shanshui/5.jpg', f'{self.predict_path}/_15.jpg')
        self.p('photo/1.jpg', f'{self.predict_path}/_16.jpg')
        self.p('photo/2.jpg', f'{self.predict_path}/_17.jpg')
        
    def gen2(self):
        self.p('culture/blue_and_white_porcelain.jpg', f'{self.predict_path}/x_1.jpg')
        self.p('culture/blue_and_white_porcelain2.jpg', f'{self.predict_path}/x_2.jpg')
        self.p('culture/blue_calico.jpg', f'{self.predict_path}/x_3.jpg')
        self.p('culture/blue_calico2.jpg', f'{self.predict_path}/x_4.jpg')
        self.p('culture/mural.jpg', f'{self.predict_path}/x_5.jpg')
        self.p('culture/ink_painting3.jpg', f'{self.predict_path}/x_6.jpg')
        self.p('culture/qingming_river_scene.jpg', f'{self.predict_path}/x_7.jpg')

with tf.device('/GPU:0'):
    main = Main()
    main.create()
    #main.load_weight()
    main.train(500)
    main.model.save(main.model_path)
    #tf.saved_model.save(main.model, main.model_path)
    #main.model.save_weights(main.path_checkpoints)
    main.gen()
    main.gen2()
    