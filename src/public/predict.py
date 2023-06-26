import os
import tensorflow as tf
import numpy as np
import math

def combine_predict(shape, list, rows, toadd):
    for i in range(toadd):
        list.append(tf.fill(shape, 1))
    
    lists = np.array_split(list, rows)
    
    tflist = []
    for element in lists:
        newlist = tf.concat(element, axis=0)
        newlist = tf.reshape(newlist, (32 * len(element), 32, 1))
        tflist.append(newlist)
        
    concatenated_tensor = tf.concat(tflist, axis=1)
    concatenated_tensor = (concatenated_tensor + 1 ) * 127.5
    concatenated_tensor = tf.where(concatenated_tensor < 100, 0, concatenated_tensor)
    return concatenated_tensor


def generate_all(model, out_path, name, dataset_train, total):
    
        side_length = int(math.sqrt(total))
        rows = math.ceil( total / side_length)
        toadd = side_length * rows - total 
    
        list_predict = []
        list_ipt = []
        for step, input_image in dataset_train.take(total).enumerate():
            prediction = model(input_image, training=False)
            list_predict.append(prediction[0])
            list_ipt.append(input_image[0])
            
        prediction_outs = combine_predict(list_predict[0].shape, list_predict, rows, toadd)
        ipt_outs = combine_predict(list_predict[0].shape, list_ipt, rows, toadd)

        splits_shape = tf.fill((32, prediction_outs.shape[1],1), tf.constant(255, dtype=tf.double))

        tf.keras.preprocessing.image.save_img(os.path.join(out_path, f'all-{name}.jpg'), tf.concat([prediction_outs, splits_shape, ipt_outs], axis=0))
        print(f'Saved checkpoint images - {name}!')
        return