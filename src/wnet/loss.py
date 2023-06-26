import tensorflow as tf

conf = {
    "label_smoothing": True,
    "alpha": 3,
    "alpha_GP": 10,
    "beta_d": 1,
    "beta_p": 0.2,
    "beta_r": 0.2,
    "lambda_l1": 50,
    "lambda_phi": 75,
    "phi_p": 3,
    "phi_r": 5
}

#平滑
class LabelSmoothing(tf.keras.losses.Loss):
    def __init__(self, smoothing=0.1, **kwargs):
        super(LabelSmoothing, self).__init__(**kwargs)
        self.smoothing = smoothing
        self.num_classes = None

    def call(self, y_true, y_pred):
        self.num_classes = y_pred.shape[-1]
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        confidence = 1.0 - self.smoothing
        smoothing_value = self.smoothing / (self.num_classes - 1)
        smooth_labels = smoothing_value * tf.ones_like(y_true) / self.num_classes
        smooth_labels = tf.where(tf.equal(y_true, 1.0), confidence - smoothing_value, smooth_labels)
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=smooth_labels)
    
#Dice Loss 是一种评估图像分割模型的损失函数，它通过计算预测值和真实值之间的交集与并集的比例来衡量预测的准确性。
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def call(self, y_true, y_pred):
        input_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        target_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])

        intersection = tf.reduce_sum(input_flat * target_flat, axis=1)
        numerator = 2.0 * intersection + self.smooth
        denominator = tf.reduce_sum(input_flat, axis=1) + tf.reduce_sum(target_flat, axis=1) + self.smooth
        loss = 1.0 - tf.reduce_mean(numerator / denominator)
        return loss
    
def calc_gradient_penalty(D, x_real, x_fake, x1, x2):
    x_real = tf.Variable(x_real, trainable=True)
    x_fake = tf.Variable(x_fake, trainable=True)
    x1 = tf.Variable(x1, trainable=True)
    x2 = tf.Variable(x2, trainable=True)
    
    alpha = tf.random.uniform(shape=x1.shape[:1], minval=0., maxval=1.)
    alpha = tf.expand_dims(tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1), axis=-1)
    interpolates = alpha * x_real + (1 - alpha) * x_fake
    interpolates = tf.cast(interpolates, tf.float32)
    
    with tf.GradientTape(watch_accessed_variables=True) as tape:
        disc_interpolates = D(interpolates, x1, x2)[:3]
    gradients = tape.gradient(disc_interpolates, [interpolates, x1, x2], 
                              output_gradients=[tf.ones_like(disc_interpolates[0]), 
                                                tf.ones_like(disc_interpolates[1]), 
                                                tf.ones_like(disc_interpolates[2])])
    
    gradient_penalty = tf.reduce_mean((tf.norm(gradients[0], ord=2, axis=1) - 1) ** 2)
    return gradient_penalty

class GenerationLoss(tf.keras.Model):
    def __init__(self):
        super(GenerationLoss, self).__init__()
        if conf.label_smoothing:
            self.cls_criterion = LabelSmoothing()
        else:
            self.cls_criterion = tf.keras.losses.SparseCategoricalCrossentropy()
        self.bce_criterion = tf.keras.losses.BinaryCrossentropy()

    def call(
        self,
        out,
        out_real,
        real_label,
        real_style_label,
        char_label,
        x_fake,
        x_real,
        encoder_out_real_left,
        encoder_out_fake_left,
        encoder_out_real_right,
        encoder_out_fake_right,
        cls_enc_p=None,
        cls_enc_s=None,
    ):
        self.real_fake_loss = conf.alpha * self.bce_criterion(
            out[0], real_label
        )
        self.style_category_loss = conf.beta_d * self.cls_criterion(
            real_style_label, out[1]
        )
        self.char_category_loss = conf.beta_d * self.cls_criterion(
            char_label, out[2]
        )

        if conf.reconstruction_loss_type == "dice":
            self.reconstruction_loss = conf.lambda_l1 * DiceLoss()(
                x_fake, x_real
            )
        elif conf.reconstruction_loss_type == "l1":
            self.reconstruction_loss = conf.lambda_l1 * tf.reduce_mean(
                tf.abs(x_fake - x_real)
            )

        # 原论文里面使用训练好的vgg字符分类网络的中间特征来做
        # 这里为了省事，直接用的Discriminator的中间层特征
        self.reconstruction_loss2 = conf.lambda_phi * (
            tf.reduce_mean((out[3][0] - out_real[3][0]) ** 2)
            + tf.reduce_mean((out[3][1] - out_real[3][1]) ** 2)
            + tf.reduce_mean((out[3][2] - out_real[3][2]) ** 2)
            + tf.reduce_mean((out[3][3] - out_real[3][3]) ** 2)
        )

        self.left_constant_loss = conf.phi_p * tf.reduce_mean(
            (encoder_out_real_left - encoder_out_fake_left) ** 2
        )
        self.right_constant_loss = conf.phi_r * tf.reduce_mean(
            (encoder_out_real_right - encoder_out_fake_right) ** 2
        )
        self.content_category_loss = conf.beta_p * self.cls_criterion(
            char_label, cls_enc_p
        )  # category loss for content prototype encoder
        self.style_category_loss = conf.beta_r * self.cls_criterion(
            real_style_label, cls_enc_s
        )
        return (
            self.real_fake_loss
            + self.style_category_loss
            + self.char_category_loss
            + self.reconstruction_loss
            + self.reconstruction_loss2
            + self.left_constant_loss
            + self.right_constant_loss
            + self.content_category_loss
            + self.style_category_loss
        )

class DiscriminationLoss(tf.keras.layers.Layer):
    def __init__(self):
        super(DiscriminationLoss, self).__init__()
        if conf.label_smoothing:
            self.cls_criteron = LabelSmoothing()
        else:
            self.cls_criteron = tf.keras.losses.CategoricalCrossentropy()

    def call(
        self,
        out_real,
        out_fake,
        real_label,
        fake_label,
        real_style_label,
        fake_style_label,
        char_label,
        fake_char_label,
        cls_enc_p=None,
        cls_enc_s=None,
        D=None,
        x_real=None,
        x_fake=None,
        x1=None,
        x2=None,
    ):
        self.real_loss = conf.alpha * tf.keras.losses.BinaryCrossentropy()(
            real_label, out_real[0]
        )  # fake or real loss
        self.fake_loss = conf.alpha * tf.keras.losses.BinaryCrossentropy()(
            fake_label, out_fake[0]
        )  # fake or real loss
        self.real_style_loss = conf.beta_d * self.cls_criteron(
            real_style_label, out_real[1]
        )  # style category loss
        self.fake_style_loss = conf.beta_d * self.cls_criteron(
            fake_style_label, out_fake[1]
        )  # style category loss
        self.real_char_category_loss = conf.beta_d * self.cls_criteron(
            char_label, out_real[2]
        )  # char category loss
        self.fake_char_category_loss = conf.beta_d * self.cls_criteron(
            fake_char_label, out_fake[2]
        )  # char category loss
        self.content_category_loss = conf.beta_p * self.cls_criteron(
            char_label, cls_enc_p
        )
        self.style_category_loss = conf.beta_r * self.cls_criteron(
            real_style_label, cls_enc_s
        )
        if D:
            self.gradient_penalty = conf.alpha_GP * calc_gradient_penalty(
                D, x_real, x_fake, x1, x2
            )
        else:
            self.gradient_penalty = 0.0

        return 0.5 * (
            self.real_loss
            + self.fake_loss
            + self.real_style_loss
            + self.fake_style_loss
            + self.real_char_category_loss
            + self.fake_char_category_loss
            + self.content_category_loss
            + self.style_category_loss
            + self.gradient_penalty
        )
