'''

from glob import glob
import os

fold_name = '../Datas/planesnet'
path = glob(os.path.join(fold_name, '*.png'))
print(path)

for imgs in range(len(path)):
    print(path[imgs][len(fold_name) + 1])
'''

from keras.datasets import mnist
from keras import layers, models
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np


class model(models.Model):

    def __init__(self):

        data = self.DATA()
        i = layers.Input(shape=data.in_shape)
        c = layers.Conv2D()

    def DATA(self):
        (in_train, out_train), (in_test, out_test) = mnist.load_data()

        in_data = np.concatenate(in_train, in_test)
        out_data = np.concatenate(out_train, out_test)

        in_train, in_test, out_train, out_test = train_test_split(in_data, out_data, test_size=0.3, random_state=0)
        