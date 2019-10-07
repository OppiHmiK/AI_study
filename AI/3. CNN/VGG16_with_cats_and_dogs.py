# -*- coding : utf-8- -*-
__author__ = 'KimHippo'

from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from keras.models import Model
import keras_sub_pack as ksp
from glob import glob
import os


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

with tf.device('/gpu:0'):
    class cat_n_dog(Model):

        def __init__(self, classes=2, in_shape=None):
            self.classes, self.in_shape = classes, in_shape
            self.model()
            i, t = self.i, self.t

            super().__init__(i, t)
            self.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        def model(self):
            classes = self.classes
            in_shape = self.in_shape

            i = Input(shape=in_shape)
            c = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(i)
            c = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)

            c = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)

            c = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)

            c = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)

            c = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)
            f = Flatten()(c)

            h = Dense(4096, activation='relu')(f)
            h = Dense(4096, activation='relu')(h)
            h = Dense(1000, activation='relu')(h)
            t = Dense(classes, activation='softmax')(h)

            self.i, self.t = i, t

    class DATA:

        def __init__(self, path, format='png', img_size=28, test_size=0.3, rand_stat=0):
            self.path = path
            self.format = format
            self.img_size = img_size

            self.preprocessing()
            in_data, tar_data = self.in_data, self.tar_data
            in_train, in_test, tar_train, tar_test = train_test_split(in_data, tar_data, test_size=test_size,
                                                                      random_state=rand_stat)
            self.in_train, self.in_test = in_train, in_test
            self.tar_train, self.tar_test = tar_train, tar_test
            self.gener = ImageDataGenerator(horizontal_flip=True, rotation_range=30, vertical_flip=True)

        def load(self):
            path, format = self.path, self.format
            img_path = glob(os.path.join(path, '*.' + format))
            labels = [img_path[label][len(path) + 1] for label in range(len(img_path))]
            images = [plt.imread(img_path[imgs]) for imgs in range(len(img_path))]
            images = np.asarray(images)
            self.images, self.labels = images, labels

        def preprocessing(self):
            self.load()
            in_data, tar_data = self.images, self.labels
            classes = len(set(tar_data))

            in_data = in_data.astype('float32')
            in_shape = in_data.shape[1:]
            tar_data = to_categorical(tar_data, classes)

            self.in_data, self.tar_data = in_data, tar_data
            self.classes, self.in_shape = classes, in_shape


    class run:

        def __init__(self, path, set_name, format='png', epo=15, batch=30, fig=1):

            data = DATA(path=path, img_size=300, format=format)
            model = cat_n_dog(classes=data.classes, in_shape=data.in_shape)
            flow = data.gener.flow(data.in_train, data.tar_train, batch_size = batch)
            histo = model.fit_generator(flow, epochs = epo, steps_per_epoch=len(data.in_train) / 32,
                                        validation_data=(data.in_test, data.tar_test))
            loss_n_accu = model.evaluate(data.in_test, data.tar_test, batch_size=batch)
            print(f'Test loss : {loss_n_accu[0]} \n accuracy : {loss_n_accu[1]}')

            if fig:
                plt.figure(figsize=(12,5))
                plot = ksp.plot(histo)
                plt.subplot(1, 2, 1)
                plot.plot_loss()

                plt.subplot(1, 2, 2)
                plot.plot_acc()
                plt.show()

            self.model = model
            self.set_name = set_name
            self.save_n_load()

        def save_n_load(self):

            model = self.model
            set_name = self.set_name
            fname = ksp.random_name()
            path = '../weights' + set_name
            if not os.path.isdir(path):
                os.mkdir(path)
            os.chdir(path)

            model.save_weights('{}_{}_kimhippo.h5'.format(fname, set_name))
            print('Train weights are saved in : ', path)


    path = '../Datas/cat_and_dog/using_model'
    set_name = 'cat_n_dog'
    format = 'png'

    run = run(path=path, set_name=set_name, format=format, epo = 50)