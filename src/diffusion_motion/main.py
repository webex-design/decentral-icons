import os
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

from keras.engine.training import _minimum_control_deps
print(_minimum_control_deps)
