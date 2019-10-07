# -*- coding : utf-8 -*-

from keras.backend import tensorflow_backend as K
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras import models, layers
import matplotlib.pyplot as plt
import plot_acc_loss as pal
import numpy as np

class GPU_DNN(models.Sequential):

    def __init__(self, Nh, Pd, No=10, Ni=None):
        self.Nh, self.Pd, self.No = Nh, Pd, No

        super().__init__()
        self.add(layers.Dense(Nh[0], input_shape=(Ni,), activation='relu', name='hidden-1'))
        self.add(layers.Dropout(Pd[0]))

        for rep in range(1, len(Nh)):
            Nh_rep, Pd_rep = Nh[rep], Pd[rep]
            self.add(layers.Dense(Nh_rep, activation='relu', name='hidden-' + str(rep+1)))
            self.add(layers.Dropout(Pd_rep))

        self.add(layers.Dense(No, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def data_load():
    (img_train, lab_train), (img_test, lab_test) = mnist.load_data()
    lab_train = to_categorical(lab_train)
    lab_test = to_categorical(lab_test)

    L, W, H = img_train.shape
    img_train = img_train.reshape(-1, W * H)
    img_test = img_test.reshape(-1, W * H)

    img_train = img_train / 255.0
    img_test  = img_test / 255.0

    return (img_train, lab_train), (img_test, lab_test)


def main(Nh, Pd, epochs=50, batch_size=200):
    (img_train, lab_train), (img_test, lab_test) = data_load()
    model = GPU_DNN(Ni=img_train.shape[1], Nh=Nh, Pd=Pd)
    histo = model.fit(img_train, lab_train, batch_size=batch_size, epochs=epochs, validation_split=0.3)
    perform_test = model.evaluate(img_test, lab_test, batch_size=batch_size)
    print('Test data Accuracy and Loss : ', perform_test)

    plot = pal.plot(histo)
    plt.subplot(1, 2, 1)
    plot.plot_acc()

    plt.subplot(1, 2, 2)
    plot.plot_loss()
    plt.show()

    return model.predict(img_test)


main(Nh = [100, 50], Pd = [0.1, 0.3], batch_size= 1000)







