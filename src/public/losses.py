import tensorflow as tf
import numpy as np
from tensorflow.image import ssim

def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def loss_ssim(y_true, y_pred):
    loss = 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))
    return loss

def loss_ssim_batch(y_true_batch, y_pred_batch):
    ssim_values = []
    for i in range(len(y_true_batch)):
        ssim_value = ssim(y_true_batch[i], y_pred_batch[i], max_val=1.0)
        ssim_values.append(ssim_value)
    mean_ssim = 1 - tf.reduce_mean(ssim_values)
    return mean_ssim

def loss_ssim_batch_shape(y_true_batch, y_pred_batch):
    ssim_values = []
    for i in range(y_true_batch.shape[0]):
        ssim_value = ssim(y_true_batch[i], y_pred_batch[i], max_val=1.0)
        ssim_values.append(ssim_value)
    mean_ssim = 1 - tf.reduce_mean(ssim_values)
    return mean_ssim

def chamfer_loss(y_true, y_pred):
    # 将输入转换为二进制图像 计算两个集合之间的平均距离，是对称性损失。
    # 计算Chamfer距离 计算一个集合到另一个集合的最大距离，是不对称性损失。
    distance_true_pred = tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)), axis=-1)
    distance_pred_true = tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)), axis=-1)

    # 损失为Chamfer距离的和
    loss = tf.reduce_mean(distance_true_pred) + tf.reduce_mean(distance_pred_true)
    return loss

def frechet_loss(y_true, y_pred):
    # 将输入转换为二进制图像 度量两个分布之间的差异性的距离指标
    # 计算Frechet距离
    distance = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))
    loss = tf.reduce_mean(distance)
    return loss

def iou_loss(y_true, y_pred):
    # 将输入转换为二进制图像 交集除以并集，用于度量分割图像与目标图像之间的重叠度
    # 计算IoU损失
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1), axis=[1, 2])
    iou = tf.reduce_mean(intersection / (union + 1e-7))

    # 损失为1 - IoU 
    loss = 1.0 - iou
    return loss

def dice_loss(y_true, y_pred):
    # 将输入转换为二进制图像 计算两倍交集除以总和，也用于度量分割图像与目标图像之间的重叠度。
    # 计算Dice系数
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = tf.reduce_mean(2.0 * intersection / (union + 1e-7))

    # 损失为1 - Dice系数
    loss = 1.0 - dice
    return loss

def manhattan_loss(y_true, y_pred):
    # 计算曼哈顿距离 ，使生成的图像能够与目标图像在像素级别上尽可能接近。
    distance = tf.reduce_sum(tf.abs(y_true - y_pred), axis=[1, 2])
    loss = tf.reduce_mean(distance)
    return loss   