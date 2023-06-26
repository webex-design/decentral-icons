import os
from cnn import Cnn
import tensorflow as tf

current_file_dir = os.path.dirname(os.path.realpath(__file__))
path_root = os.path.abspath(os.path.join(current_file_dir, '../../'))
path_data = os.path.join(path_root, 'data/output/regular')

save_path = os.path.join(path_root, 'data/models/regular_cnn/my_module.h5')

cnn = Cnn(path_data)

print(cnn.num_label)
with tf.device('/GPU:0'):
    cnn.model_fn()
    cnn.model.summary()
    cnn.input_fn()
    cnn.fit(100)
    cnn.save(save_path)