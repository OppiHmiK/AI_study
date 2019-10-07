# -*- coding : utf-8 -*-
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn import model_selection
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Model
import keras_sub_pack as skp
import keras.backend as K
import tensorflow as tf
import numpy as np
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

with tf.device('/gpu:0'):

    class learning(Model):

        def __init__(self, Conv, Nh, Pd, classes, in_shape):
            self.classes, self.in_shape = classes, in_shape
            self.Conv, self.Nh, self.Pd = Conv, Nh, Pd

            super().__init__()
            self.build_model()
            super().__init__(self.i, self.o)
            self.compile()

        def build_model(self):

            classes, in_shape = self.classes, self.in_shape
            Conv, Nh, Pd = self.Conv, self.Nh, self.Pd

            # NOTE : Convolution layer
            i = Input(in_shape)
            h = Conv2D(Conv[0], kernel_size = (3,3), activation = 'relu')(i)
            for rep in range(1, len(Conv)):
                h = Conv2D(Conv[rep], kernel_size=(3, 3), activation='relu')(h)
                h = MaxPooling2D(pool_size=(2, 2))(h)
            h = Dropout(0.25)(h)
            h = Flatten()(h)

            z_cl = h

            # NOTE : Fully Connected layer
            for rep in range(len(Nh)):
                Nh_rep, Pd_rep = Nh[rep], Pd[rep]
                h = Dense(Nh_rep, activation = 'relu')(h)
                h = Dropout(Pd_rep)(h)

            o = Dense(classes, activation='softmax')(h)
            z_fl = h

            self.model_cl = Model(i, z_cl)
            self.model_fl = Model(i, z_fl)

            self.i, self.o = i, o

        def compile(self):
            Model.compile(self, loss = 'categorical_crossentropy', optimizer='adadelta', metrics = ['accuracy'])

    class DATA:

        def __init__(self, in_data, out_data, classes, test_size = 0.2, rand_stat = 0, scaling = True):

            self.in_data = in_data
            self.add_channels()

            in_data = self.in_data
            in_train, in_test, out_train, out_test = model_selection.train_test_split(in_data, out_data, test_size = test_size, random_state = rand_stat)
            in_train = in_train.astype('float32')
            in_test = in_test.astype('float32')

            if scaling:

                scaler = MinMaxScaler()
                In_shape = in_train.shape[0]
                in_train = scaler.fit_transform(in_train.reshape(In_shape, -1)).reshape(in_train.shape)
                In_shape = in_test.shape[0]
                in_test = scaler.transform(in_test.reshape(In_shape, -1)).reshape(in_test.shape)
                self.scaler = scaler

            out_train = to_categorical(out_train, classes)
            out_test = to_categorical(out_test, classes)

            self.in_train, self.out_train = in_train, out_train
            self.in_test, self.out_test = in_test, out_test

        def add_channels(self):

            in_data = self.in_data

            if len(in_data.shape) == 3:

                length, width, height = in_data.shape
                if K.common.image_dim_ordering() == 'th':
                    in_data = in_data.reshpe(length, 1, width, height)
                    in_shape = (1, width, height)

                else:
                    in_data = in_data.reshape(length, width, height, 1)
                    in_shape = (width, height, 1)

            else:
                in_shape = in_data.shape[1:]


            self.in_data = in_data
            self.in_shape = in_shape

    class machine:

        def __init__(self, in_data, out_data, Conv, Nh, Pd, classes = 2, fig = True):

            self.classes = classes
            self.fig = fig

            self.Conv, self.Nh, self.Pd = Conv, Nh, Pd

            self.set_data(in_data, out_data)
            self.set_model()

        def set_data(self, in_data, out_data):
            classes = self.classes
            self.data = DATA(in_data, out_data, classes)

        def set_model(self):

            Conv, Nh, Pd = self.Conv, self.Nh, self.Pd
            classes = self.classes
            data = self.data

            self.model = learning(Conv, Nh, Pd, classes, data.in_shape)

        def fit(self):

            data = self.data
            model = self.model
            epo, batch = self.epo, self.batch

            # early = EarlyStopping(monitor = 'val_acc', mode = 'auto')
            histo = model.fit(data.in_train, data.out_train, epochs = epo,
                              batch_size = batch, validation_data = (data.in_test, data.out_test))
            return histo

        def prediction(self, smaples, pred_opt = True):

            if pred_opt:
                pred = self.run()
                data = self.data
                for rep in smaples:
                    pred_ind = np.argmax(pred[rep])
                    exact_ind = np.argmax(data.out_test[rep])
                    print(f'Prediction index : {pred_ind} \n Exact index : {exact_ind}')

        def save_n_load(self):

            suffix = skp.unique_filename('datatime')
            fname = 'weights_'+suffix
            model = self.model

            os.chdir('./weights')
            os.chdir('DCNN_weights')
            model.save_weights(suffix + '.h5')
            print('Weights are saved in ', fname)

        def run(self, epo = 15, batch = 128):

            data = self.data
            model = self.model
            fig = self.fig
            self.epo, self.batch = epo, batch

            epo, batch = self.epo, self.batch
            histo = self.fit()
            loss_n_accu = model.evaluate(data.in_test, data.out_test, batch_size = batch)

            print(f'Test loss : {loss_n_accu[0]} \n accuracy : {loss_n_accu[1]}')

            self.save_n_load()
            if fig:
                plot = skp.plot(histo)
                plt.figure(figsize=(12, 5))

                plt.subplot(1,2,1)
                plot.plot_acc()

                plt.subplot(1,2,2)
                plot.plot_loss()
                plt.show()

            self.histo = histo
            model.summary()
            return model.predict(data.in_test, batch_size = batch)
