# -*- coding  : utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import backend as K
import keras_sub_pack as ksp
from keras import models
import numpy as np

config = K.tensorflow_backend.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tensorflow_backend.tf.Session(config = config)

class mnist_CNN(models.Sequential):

    def __init__(self, in_shape, num_classes):

        super().__init__()
        # NOTE : hidden layer 1
        self.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape = in_shape))

        # NOTE : hidden layer 2
        self.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
        self.add(MaxPooling2D(pool_size = (2, 2)))
        self.add(Dropout(0.25))
        self.add(Flatten())

        # NOTE : Fully Connected
        self.add(Dense(128, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

class DATA():

    def __init__(self):

        num_classes = 10

        (img_train, lab_train), (img_test, lab_test) = mnist.load_data()
        width, height = img_train.shape[1:]

        if K.image_data_format() == 'channels_first':
            img_train = img_train.reshape(img_train.shape[0], 1, width, height)
            img_test = img_test.reshape(img_test.shape[0], 1, width, height)
            input_shape = (1, width, height)

        else:
            img_train = img_train.reshape(img_train.shape[0], width, height, 1)
            img_test = img_test.reshape(img_test.shape[0], width, height, 1)
            input_shape = (width, height, 1)

        img_train = img_train.astype('float32')
        img_test = img_test.astype('float32')

        img_train = img_train / 255.0
        img_test = img_test / 255.0

        lab_train = to_categorical(lab_train, num_classes)
        lab_test = to_categorical(lab_test, num_classes)

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.img_train, self.lab_train = img_train, lab_train
        self.img_test, self.lab_test = img_test, lab_test

class main():

    def __init__(self, Epochs = 50, Batch_size = 1000):

        self.Epochs, self.Batch_size = Epochs, Batch_size
        data = DATA()
        model = mnist_CNN(in_shape = data.input_shape, num_classes = data.num_classes)
        histo = model.fit(data.img_train, data.lab_train, epochs = Epochs, batch_size = Batch_size, validation_split=0.3)
        score = model.evaluate(data.img_test, data.lab_test, batch_size=Batch_size)
        print(f'Test data Accuracy : {score[0]}, Loss : {score[1]}')
        print(model.summary())

        self.pred = model.predict(data.img_test)
        self.histo = histo

    def draw_plot(self):

        plot = ksp.plot(self.histo)
        plt.subplot(1,2,1)
        plot.plot_acc()

        plt.subplot(1,2,2)
        plot.plot_loss()
        plt.show()

    def prediction(self, arr):

        data = DATA()
        predict = main(self.Epochs, self.Batch_size)
        for rep in arr:
            pred_ind, exact_ind = np.argmax(predict.pred[rep]), np.argmax(data.lab_test[rep])
            print(f'Prediction : {pred_ind} \n Exact : {exact_ind}')

if __name__ == '__main__':
    m = main(Epochs = 10, Batch_size=2000)

    np.random.seed(999)
    rand = np.random.randint(1, 10000, 10)
    m.prediction(rand)








