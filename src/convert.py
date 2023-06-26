import tensorflow as tf
import os

current_file_path = os.path.realpath(__file__)
current_file_dir = os.path.dirname(current_file_path)
from_path = os.path.abspath(os.path.join(current_file_dir, '../models/vgg-19/py/magenta_arbitrary-image-stylization-v1-256_2/saved_model.pb'))
to_path = os.path.abspath(os.path.join(current_file_dir, '../models/vgg-19/js/stylization-js'))

# 加载原始的 .pb 模型
graph_def = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile(from_path, 'rb') as f:
    graph_def.ParseFromString(f.read())

# 将模型转换为 SavedModel 格式
with tf.compat.v1.Session() as sess:
    tf.import_graph_def(graph_def, name='')
    tf.saved_model.save(sess, to_path)