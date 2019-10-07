# -*- coding : utf-8 -*-

from keras import models, layers
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras import datasets
import numpy as np

class ANN(models.Model):
    def __init__(self, Ni, Nh, No):
        self.Ni = Ni
        self.Nh = Nh
        self.No = No

        hidden = layers.Dense(self.Nh)
        out = layers.Dense(self.No)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        i = layers.Input(shape = (self.Ni, ))
        h = hidden(relu(i))
        o = softmax(out(h))

        super().__init__(i, o)
        self.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def data_load(self):
        (in_train, out_train), (in_test, out_test) = datasets.mnist.load_data()
        out_train, out_test = np_utils.to_categorical(out_train), np_utils.to_categorical(out_test)
        W, H = in_train.shape[1], in_train.shape[2]
        in_train, in_test = in_train.reshape(-1, W*H), in_test.reshape(-1, W*H)
        in_train, in_test = in_train / 255.0, in_test / 255.0
        return (in_train, out_train), (in_test, out_test)

def main():
    Ni = 784; Nh = 100; No = 10
    model = ANN(Ni, Nh, No)
    (in_train, out_train), (in_test, out_test) = model.data_load()
    history = model.fit(in_train, out_train, epochs = 15,
                        batch_size = 100, validation_split= 0.3)

    perform_test = model.evaluate(in_test, out_test, batch_size= 100)
    print('Test loss and Accuracy : ', perform_test)

main()