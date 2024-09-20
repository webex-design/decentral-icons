import os
import re
import shutil
import tensorflow as tf
import tensorflow_hub as hub

current_file_dir = os.path.dirname(os.path.realpath(__file__))
path_root = os.path.abspath(os.path.join(current_file_dir, '../../'))
path_model = os.path.join(path_root, 'models/magenta_arbitrary-image-stylization-v1-256_2')
path_style = os.path.join(path_root, 'data/style/art')
path_content = os.path.join(path_root, 'data/scene/Clarissa_Smith.png')
path_output = os.path.join(path_root, 'models/vgg/output')

default_re = re.compile(r'.(jpg|png)$', re.I)

class VGG:
    def __init__(self):
        self.hub_module = hub.load(path_model)
        return
    
    def __scan(self, path, filter=default_re):
        files = os.listdir(path)
        ret = []
        for x in files:
            _path = os.path.join(path, x)
            if os.path.isdir(_path):
                ret += self.__scan(path)
            elif filter==None or re.search(filter, x)!=None:
                ret.append(_path)
        return ret
    
    def __load(self, path, need_resize=True):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, 3)
        if need_resize:
            image = tf.image.resize(image, [256,256]) # [256,256]
        return tf.cast(image / 255, tf.float32), path
    
    def create_dataset_one(self, sources):
        dataset_train = tf.data.Dataset.from_tensor_slices(sources)
        dataset_train = dataset_train.map(self.__load)
        #dataset_train = dataset_train.shuffle(buffer_size=len(sources))
        dataset_train = dataset_train.batch(batch_size=1)
        return dataset_train
    
    def predict_img(self, content, style, name):
        outputs = self.hub_module(content, style)
        stylized_image = outputs[0] * 255
        print(f'generate {name}.jpg')
        tf.keras.preprocessing.image.save_img(os.path.join(path_output, f'{name}.jpg'), stylized_image[0])
    
    def run(self):
        content, content_file_name = self.__load(path_content, False)
        content = content[tf.newaxis, :]

        style_paths = self.__scan(path_style)
        my_dataset = self.create_dataset_one(style_paths)
        
        i=0
        if os.path.exists(path_output):
            shutil.rmtree(path_output)
        if not os.path.exists(path_output):
            os.makedirs(path_output)
            
        for style_image,file_name in my_dataset:
            self.predict_img(content, style_image, os.path.splitext(os.path.basename(file_name[0].numpy().decode()))[0])
            i+=1
        return
    
with tf.device('/GPU:0'):
    vgg = VGG()
    vgg.run()