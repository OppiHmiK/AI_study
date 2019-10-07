# -*- coding : utf-8- -*-
__author__ = 'KimHippo'

from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from keras import models
import keras_sub_pack as ksp
from glob import glob
import keras
import os
from keras.applications.imagenet_utils import preprocess_input


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

with tf.device('/gpu:0'):
    class cat_n_dog(models.Model):

        def __init__(self, classes=2, in_shape=None):
            self.classes, self.in_shape = classes, in_shape
            self.model()
            i, t = self.i, self.t

            super().__init__(i, t)
            opti = keras.optimizers.SGD(lr = 0.01, momentum = 0.9, nesterov=True)
            self.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])

        def model(self):
            classes = self.classes
            in_shape = self.in_shape

            i = Input(shape=in_shape)
            c = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(i)
            c = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)
            c = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)
            c = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)
            c = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(c)
            c = MaxPooling2D(pool_size=(2, 2))(c)
            c = Dropout(0.25)(c)
            f = Flatten()(c)

            h = Dense(50, activation='relu')(f)
            h = Dropout(0.05)(h)
            h = Dense(1000,activation='relu')(h)
            h = Dropout(0.2)(h)
            h = Dense(150, activation='relu')(h)
            h = Dropout(0.15)(h)
            h = Dense(400, activation='relu')(h)
            h = Dropout(0.2)(h)
            h = Dense(1500, activation='relu')(h)
            h = Dropout(0.15)(h)
            h = Dense(500, activation='relu')(h)
            h = Dropout(0.2)(h)
            h = Dense(2000, activation='relu')(h)
            h = Dropout(0.15)(h)
            h = BatchNormalization()(h)
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
            labels = [int(img_path[label][len(path) + 1]) for label in range(len(img_path))]
            images = [plt.imread(img_path[imgs]) for imgs in range(len(img_path))]
            images = np.asarray(images)
            images = preprocess_input(images)
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

        def __init__(self, path, set_name, format='png', steps_per_epo = 15, epo=15, batch=30, fig=1):

            data = DATA(path=path, img_size=300, format=format)
            model = cat_n_dog(classes=data.classes, in_shape=data.in_shape)

            path = '../weights/'+set_name
            weights_name = '/{}_kimhippo.h5'.format(set_name)

            reduce = callbacks.ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1, patience=100)
            check_point = callbacks.ModelCheckpoint(filepath = path + weights_name, monitor = 'val_acc', save_best_only = True)

            callback = [check_point, reduce]
            flow = data.gener.flow(data.in_train, data.tar_train, batch_size = batch)
            histo = model.fit_generator(flow, epochs = epo, steps_per_epoch= steps_per_epo, callbacks=callback,
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
            path = '../weights/' + set_name
            if not os.path.isdir(path):
                os.mkdir(path)
            os.chdir(path)

            model.save_weights('_{}_kimhippo.h5'.format(set_name), overwrite = True)
            print('Train weights are saved in : ', path)


    path = '../Datas/cat_and_dog/using_model'
    set_name = 'cat_n_dog'
    format = 'png'

    run = run(path=path, set_name=set_name, format=format, epo = 1000, steps_per_epo = 150)