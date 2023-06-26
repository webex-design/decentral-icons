import os
import re
import json
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

default_re = re.compile(r'.(jpg|png)$', re.I)
name_re = re.compile(r'x4.', re.I)

class Super_Dataset:
    def __init__(self, train_rate =0.8):
        self.len = 1
        self.batch_size = 16
        self.train_rate = train_rate
        return
    
    # Normalizing the images [0,255] to [-1, 1]
    def normalize(self, image):
        return image / 255
    
    def unnormalize(self, image):
        return image * 255
    
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
    
    def load(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3 ,dtype=tf.dtypes.float32)
        #image = self.normalize(image)
        return image
    
    def flip_left_right(self, lowres_img, highres_img):
        """Flips Images to left and right."""

        # Outputs random values from a uniform distribution in between 0 to 1
        rn = tf.random.uniform(shape=(), maxval=1)
        # If rn is less than 0.5 it returns original lowres_img and highres_img
        # If rn is greater than 0.5 it returns flipped image
        return tf.cond(
            rn < 0.5,
            lambda: (lowres_img, highres_img),
            lambda: (
                tf.image.flip_left_right(lowres_img),
                tf.image.flip_left_right(highres_img),
            ),
        )

    def random_rotate(self, lowres_img, highres_img):
        """Rotates Images by 90 degrees."""

        # Outputs random values from uniform distribution in between 0 to 4
        rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        # Here rn signifies number of times the image(s) are rotated by 90 degrees
        return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)
        

    def random_crop(self, lowres_img, highres_img, hr_crop_size=96, scale=4):
        """Crop images.

        low resolution images: 24x24
        high resolution images: 96x96
        """
        lowres_crop_size = hr_crop_size // scale  # 96//4=24
        lowres_img_shape = tf.shape(lowres_img)[:2]  # (height,width)

        lowres_width = tf.random.uniform(
            shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
        )
        lowres_height = tf.random.uniform(
            shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
        )

        highres_width = lowres_width * scale
        highres_height = lowres_height * scale

        lowres_img_cropped = lowres_img[
            lowres_height : lowres_height + lowres_crop_size,
            lowres_width : lowres_width + lowres_crop_size,
        ]  # 24x24
        highres_img_cropped = highres_img[
            highres_height : highres_height + hr_crop_size,
            highres_width : highres_width + hr_crop_size,
        ]  # 96x96
        lowres_img_cropped = tf.ensure_shape(lowres_img_cropped, (lowres_crop_size, lowres_crop_size, 3))
        highres_img_cropped = tf.ensure_shape(highres_img_cropped, (hr_crop_size, hr_crop_size, 3))
        return lowres_img_cropped, highres_img_cropped
    
    @tf.autograph.experimental.do_not_convert
    def load_images(self, low_img_p, high_img_p):
        high_img = self.load(high_img_p)
        low_img = self.load(low_img_p)
        return low_img, high_img
    
    def reg_high_name(self, low):
        return re.sub(name_re, '.', low)
    
    def read(self, low_paths, high_paths):
        high_images = []
        low_images = []
        phigh = self.__scan(high_paths)
        plow = self.__scan(low_paths)
        for _p_low in plow:
            _p_high = self.reg_high_name(_p_low.replace(low_paths, high_paths))
            if _p_high in phigh:
                high_images.append(_p_high)
                low_images.append(_p_low)
        self.len = len(high_images)
        self.source = (low_images, high_images)
        return self.source
    
    def create_dataset(self, training=True):
        dataset_train = tf.data.Dataset.from_tensor_slices(self.source)
        dataset_train = dataset_train.map(self.load_images)
        train_num = int(self.len * self.train_rate)
        
        ds_train = dataset_train.take(train_num)
        ds_val = dataset_train.skip(train_num)
        
        ds_train = self.__creat_dataset(ds_train, training)
        ds_val = self.__creat_dataset(ds_val, training)
        
        return ds_train, ds_val
    
    def __creat_dataset(self, ds_train, training=True):
        ds = ds_train.map(
            lambda lowres, highres: self.random_crop(lowres, highres, scale=4),
            num_parallel_calls=AUTOTUNE,
        )
        if training:
            ds = ds.map(self.random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(self.flip_left_right, num_parallel_calls=AUTOTUNE)
        # Batching Data
        ds = ds.batch(self.batch_size)

        if training:
            # Repeating Data, so that cardinality if dataset becomes infinte
            ds = ds.repeat(1)
        # prefetching allows later images to be prepared while the current image is being processed
        ds = ds.prefetch(buffer_size=AUTOTUNE).cache()
        return ds