# -*- coding : utf-8 -*-

from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras import models, layers
import matplotlib.pyplot as plt
import plot_acc_loss as pal
import numpy as np

class color_DNN(models.Sequential):

    def __init__(self, Nh, Pd, No = 10, Ni = None):

        super().__init__()
        self.add(layers.Dense(Nh[0], input_shape = (Ni, ), activation = 'relu'))
        self.add(layers.Dropout(Pd[0]))
        for rep in range(1, len(Nh)):
            Nh_rep, Pd_rep = Nh[rep], Pd[rep]
            self.add(layers.Dense(Nh_rep, activation = 'relu'))
            self.add(layers.Dropout(Pd_rep))
        self.add(layers.Dense(No, activation='softmax'))
        self.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

def data_load():

    (img_train, lab_train), (img_test, lab_test) = cifar10.load_data()

    lab_train = to_categorical(lab_train)
    lab_test = to_categorical(lab_test)

    L, W, H, C = img_train.shape
    img_train = img_train.reshape(-1, W*H*C)
    img_test = img_test.reshape(-1, W*H*C)

    img_train = img_train / 255.0
    img_test = img_test / 255.0
    return (img_train, lab_train), (img_test, lab_test)

def main(Nh, Pd, epochs = 20):
    (img_train, lab_train), (img_test, lab_test) = data_load()
    model = color_DNN(Ni=img_train.shape[1], Nh = Nh, Pd = Pd, No = 10)
    histo = model.fit(img_train, lab_train, batch_size=2000, epochs = epochs, validation_split= 0.2)
    perform_test = model.evaluate(img_test, lab_test, batch_size=2000)

    plot = pal.plot(histo)
    plt.subplot(1,2,1)
    plot.plot_acc()
    plt.subplot(1,2,2)
    plot.plot_loss()
    print('Test Accuracy and Loss : ', perform_test)

    return model.predict(img_test)

def prediction(args, Nh, Pd, epochs = 20):

    (_, _), (_, lab_test) = data_load()
    pred = main(Nh, Pd, epochs)
    for rep in args:
        pred_ind = np.argmax(pred[rep])
        exact_ind = np.argmax(lab_test[rep])

        print('Prediction index : ', pred_ind)
        print('Exact index : ', exact_ind)



if __name__ == '__main__':
    np.random.seed(999)
    rand = np.random.randint(1, 10000, 10)
    prediction(rand, Nh = [50, 100, 200], Pd = [0.0, 0.0, 0.0], epochs = 150)

