import os
from libs.data import builder
import tensorflow as tf

#command_path = os.path.realpath(sys.argv[0])
current_file_path = os.path.realpath(__file__)
current_file_dir = os.path.dirname(current_file_path)
source_regular = os.path.abspath(os.path.join(current_file_dir, '../test/test1'))
output_regular = os.path.abspath(os.path.join(current_file_dir, '../test/test2'))
config = None # os.path.join(source_regular, 'data.json')

def gen():
    myBuilder = builder.Builder()
    myBuilder.read(source_regular, config).load()
    myBuilder.rotate(5, 340, 10)
    myBuilder.save(output_regular)
    #print(myBuilder.images[0][0].numpy().tolist())
    return

with tf.device('/GPU:0'):
    gen()

print('Finished!')
