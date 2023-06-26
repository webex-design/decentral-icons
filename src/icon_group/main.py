import os
import tensorflow as tf
from icon_group import icon_group
from combine import Combine

current_file_dir = os.path.dirname(os.path.realpath(__file__))
path_root = os.path.abspath(os.path.join(current_file_dir, '../../'))
path_source = os.path.join(path_root, 'data/source/icon-data')
path_source_config = os.path.join(path_source, 'data.json')
path_source_config = os.path.join(path_source, 'data.json')
path_predict = os.path.join(path_root, 'output', 'icon_types')
path_saved_data = os.path.join(path_root, 'output', 'icon_types_data')

def run():
    icon_group.conf(path_saved_data)
    presentors, groups = icon_group.hierarchical_clustering(path_source, path_source_config)
    print('[STEP]...presentors:', presentors)
    myc = Combine(path_predict)
    i=0
    for key,values in sorted(groups.items(), key=lambda x: -len(x[1])) :
        i = i+1
        print(values)
        myc.combine(path_source, path_source_config, values, presentors[key], i)

with tf.device('/GPU:0'):
    run()
