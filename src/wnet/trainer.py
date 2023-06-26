import tensorflow as tf
from tensorflow import keras
from .loss import LabelSmoothing, DiceLoss, GenerationLoss, DiscriminationLoss
from .model import ConvBNRelu, DeConvBNRelu, ResBlock, Encoder, WNet, Discriminator

class Trainer():
    def __init__():
        return
    
    def train_one_epoch(self, epoch):
        return
    
    def init_model(self):
        self.G = WNet(5)
        self.D = Discriminator()
        return
    
    def save_model(self):
        return
    
    def eval_model(self):
        return