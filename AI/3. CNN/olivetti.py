# -*- coding : utf-8 -*-

from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping
from keras import models
from keras.utils.np_utils import to_categorical
import keras_sub_pack as ksp
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

with tf.device('/gpu:0'):

    class olivetti_CNN(models.Model):

        def __init__(self, conv, nh, pd, in_shape, classes):

            i = Input(shape = in_shape)
            c = Conv2D(conv[0], kernel_size = (3,3), activation='relu')(i)
            for rep in range(1, len(conv)):
                c = Conv2D(conv[rep], kernel_size = (3,3), activation='relu')(c)
                c = MaxPooling2D(pool_size = (2,2))(c)
            c = Dropout(pd[0])(c)
            c = Flatten()(c)

            h = Dense(nh[0], activation='relu')(c)
            h = Dropout(pd[1])(h)

            for rep in range(1, len(nh)):
                h = Dense(nh[rep], activation='relu')(h)
                h = Dropout(pd[rep])(h)
            o = Dense(classes, activation='softmax')(h)

            super().__init__(i, o)
            self.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

    class DATA:

        def __init__(self, rand_stat = 0, test_size = 0.3):

            faces = fetch_olivetti_faces()
            self.in_data, self.out_data = faces.images, faces.target

            self.preprocessing()
            in_data, out_data = self.in_data, self.out_data

            in_train, in_test, out_train, out_test = train_test_split(in_data, out_data, random_state=rand_stat, test_size = test_size)
            self.in_train, self.out_train = in_train, out_train
            self.in_test, self.out_test = in_test, out_test
            self.generator = ImageDataGenerator(rotation_range = 20, width_shift_range=0.05, height_shift_range=0.05)

        def preprocessing(self):

            in_data, out_data = self.in_data, self.out_data
            classes = len(set(out_data))

            length, width, height = in_data.shape
            if K.common.image_dim_ordering() == 'th':
                in_data = in_data.reshape(length, 1, width, height)
                in_shape = (1, width, height)

            else:
                in_data = in_data.reshape(length, width, height, 1)
                in_shape = (width, height, 1)

            out_data = to_categorical(out_data, classes)
            in_data = in_data.astype('float32')

            self.in_data, self.out_data = in_data, out_data
            self.in_shape, self.classes = in_shape, classes

    class main:

        def __init__(self, conv, nh, pd, epo = 100, batch = 20, test_size = 0.3):
            data = DATA(test_size = test_size)
            self.batch, self.epo = batch,epo
            self.data = data
            self.model = olivetti_CNN(conv = conv, nh = nh, pd = pd, in_shape = data.in_shape, classes = data.classes)

            gener = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
            early = EarlyStopping(monitor = 'val_loss', min_delta = 0, mode = 'auto', patience=25)
            flow = gener.flow(data.in_train, data.out_train, batch_size = batch)

            # histo = self.model.fit(data.in_train, data.out_train, batch_size = batch
            #                      callbacks=[early], epochs = epo)
            histo = self.model.fit_generator(flow, steps_per_epoch = 100, epochs = epo, validation_data = (data.in_test, data.out_test))

            loss_n_accu = self.model.evaluate(data.in_test, data.out_test, batch_size=batch)
            print(f'Test loss : {loss_n_accu[0]} \n accuracy : {loss_n_accu[1]}')

            plot = ksp.plot(histo)
            plt.figure(figsize=(12, 5))
            plt.subplot(1,2,1)
            plot.plot_loss()

            plt.subplot(1,2,2)
            plot.plot_acc()
            plt.show()

            pred = self.model.predict(data.in_test, batch_size = batch)
            self.save_n_load()
            self.pred = pred


        def save_n_load(self):

            model = self.model
            suffix = ksp.random_name()

            os.chdir('../weights')
            if not os.path.isdir('olivetti_CNN'):
                os.mkdir('olivetti_CNN')
            os.chdir('olivetti_CNN')

            model.save_weights(suffix + '_kimhippo.h5')
            print('Weights are save in ', os.getcwd())

        def prediction(self, samples):

            pred, data = self.pred, self.data
            for rep in samples:
                pred_ind = np.argmax(pred[rep])
                exac_ind = np.argmax(data.out_test[rep])

                print(f'Predict index : {pred_ind} \n Exact index : {exac_ind}')


if __name__ == '__main__':
    conv = [64, 128, 32]
    nh = [50, 100, 25]
    pd = [0.1, 0.2, 0.05]

    run = main(conv = conv, nh = nh, pd = pd, epo = 100, batch = 10, test_size = 0.2)

    np.random.seed(999)
    samples = np.random.randint(1, 20, 10)
    run.prediction(samples)












