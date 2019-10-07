# -*- coding : utf-8 -*-
# __author__ : KimHippo

from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras import callbacks
from keras.utils.np_utils import to_categorical
from matplotlib.image import imread
import matplotlib.pyplot as plt
from keras.models import Model
import keras_sub_pack as ksp
import tensorflow as tf
import numpy as np
import glob
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.device('/gpu:0'):

    class model(Model):

        def __init__(self, conv, nh, pd, in_shape, classes = 2):

            self.conv, self.nh, self.pd = conv, nh, pd
            self.classes = classes
            self.in_shape = in_shape

            self.build_model()
            i, o = self.i, self.o

            super().__init__(i, o)
            self.compile(loss = 'binary_crossentropy', optimizer = 'adadelta', metrics=['accuracy'])

        def build_model(self):

            conv, nh, pd = self.conv, self.nh, self.pd
            in_shape = self.in_shape
            classes = self.classes

            i = Input(shape = in_shape)
            c = Conv2D(conv[0], kernel_size = (2,2), activation ='relu')(i)
            for convs in range(1, len(conv)):
                c = Conv2D(conv[convs], kernel_size = (2,2), activation = 'relu')(c)
                c = MaxPooling2D(pool_size = (2,2))(c)
            c = Dropout(pd[0])(c)
            c = Flatten()(c)

            h = Dense(nh[0], activation = 'relu')(c)
            for nodes in range(1, len(nh)):
                h = Dense(nh[nodes], activation = 'relu')(h)
                h = Dropout(pd[nodes])(h)
            o = Dense(classes, activation = 'softmax')(h)
            self.i, self.o = i, o

    class DATA:

        def __init__(self, fold_name, test_size = 0.2, rand_stat = 0):

            self.fold_name = fold_name
            self.preprocessing()
            in_data, out_data = self.in_data, self.out_data

            in_train, in_test, out_train, out_test = train_test_split(in_data, out_data, test_size = test_size, random_state = rand_stat)
            self.in_train, self.out_train = in_train, out_train
            self.in_test, self.out_test = in_test, out_test

        def load(self):
            fold_name = self.fold_name
            img_path = glob.glob(os.path.join(fold_name, '*.png'))
            labels = [img_path[imgs][len(fold_name) + 1] for imgs in range(len(img_path))]
            images = [imread(file) for file in img_path]
            images = np.asarray(images)
            self.images, self.labels = images, labels

        def preprocessing(self):
            self.load()
            in_data, out_data = self.images, self.labels

            classes = len(set(out_data))
            in_shape = in_data.shape[1:]

            out_data = to_categorical(out_data, classes)
            in_data = in_data.astype('float32')

            self.in_data, self.out_data = in_data, out_data
            self.in_shape, self.classes = in_shape, classes

    class main:

        def __init__(self, fold_name, set_name, conv, nh, pd, epo = 20, batch = 256, fig = 1):

            datas = DATA(fold_name)
            m = model(conv, nh, pd, datas.in_shape, datas.classes)
            path = '../weights/'+set_name
            weights_name = '/{}_kimhippo.h5'.format(set_name)
            reduce = callbacks.ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1)
            check_point = callbacks.ModelCheckpoint(filepath = path + weights_name, monitor = 'val_acc', save_best_only = True)

            callback = [check_point, reduce]
            histo = m.fit(datas.in_train, datas.out_train, batch_size = batch, callbacks=callback,
                          epochs = epo, validation_split=0.3)
            loss_n_accu = m.evaluate(datas.in_test, datas.out_test, batch_size = batch)
            print(f'Test loss : {loss_n_accu[0]} \n accuracy : {loss_n_accu[1]}')

            if fig:
                plot = ksp.plot(histo)
                plt.subplot(1,2,1)
                plot.plot_loss()

                plt.subplot(1,2,2)
                plot.plot_acc()
                # plt.show()

            self.prediction = m.predict(datas.in_test, batch_size = batch)
            self.m, self.set_name  = m, set_name

            self.save_n_load()

        def save_n_load(self):

            m = self.m
            set_name = self.set_name
            path = '../weights/' + set_name
            if not os.path.isdir(path):
                os.mkdir(path)
            os.chdir(path)

            m.save_weights(f'{set_name}_kimhippo.h5')
            weights_name = '{}_kimhippo.h5'.format(set_name)
            print('Train weights are saved in : {} as {}'.format(path, weights_name))



data_dir = '../Datas/planesnet'
set_name = '_planesnet'
conv = [32, 128, 64]
nh = [50, 200, 150]
pd = [0.05, 0.2, 0.15]
main(data_dir, set_name, conv, nh, pd)