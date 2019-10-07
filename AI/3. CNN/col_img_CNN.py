# -*- coding : utf-8 -*-
# NOTE : using LeNet

from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection, metrics
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
import keras_sub_pack as ksp
import tensorflow as tf
import numpy as np
import os

assert K.image_data_format() == 'channels_last'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config = config)

with tf.device('/gpu:0'):

    # NOTE : 학습 모델 클래스
    class CNN(Model):

        def __init__(model, classes, in_shape=None):
            super().__init__()

            model.classes = classes
            model.in_shape = in_shape

            model.build_model()
            super().__init__(model.i, model.o)
            model.compile()

        def build_model(model):
            classes = model.classes
            in_shape = model.in_shape

            # NOTE : Convolution layer (Conv layer)
            i = Input(in_shape)
            h = Conv2D(32, kernel_size=(3, 3), activation='relu')(i)
            h = Conv2D(64, kernel_size=(3, 3), activation='relu')(h)
            h = MaxPooling2D(pool_size=(2, 2))(h)
            h = Dropout(0.25)(h)
            h = Flatten()(h)
            z_cl = h

            # NOTE : Fully Connected layer (FC layer)
            h = Dense(128, activation='relu')(h)
            h = Dropout(0.5)(h)
            z_fl = h

            o = Dense(classes, activation='softmax', name='preds')(h)

            # NOTE : Conv layer와 FC layer의 성능을 각각 평가하기 위해 모델을 각각 만듦.
            model.cl_part = Model(i, z_cl)
            model.fl_part = Model(i, z_fl)

            model.i, model.o = i, o

        def compile(model):
            Model.compile(model, loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # NOTE : 데이터 전처리 클래스
    class DATA:

        def __init__(self, img, lab, classes, test_size = 0.2, rand_stat = 0, scaling = True):

            self.img = img
            self.add_channels()

            img = self.img
            # NOTE : train dataset과 test dataset을 병합하여 무작위로 섞은뒤 8:2의 비율로 분할함.
            Img_train, Img_test, lab_train, lab_test = model_selection.train_test_split(img, lab, test_size = test_size, random_state=rand_stat)
            print(Img_train.shape, lab_train.shape)

            Img_train = Img_train.astype('float32')
            Img_test = Img_test.astype('float32')

            # NOTE : scaling이 True인 경우 이미지의 최댓값, 최솟값을 특정한 값이 되도록 조정
            if scaling:
                scaler = MinMaxScaler()
                img_shape = Img_train.shape[0]
                Img_train = scaler.fit_transform(Img_train.reshape(img_shape, -1)).reshape(Img_train.shape)
                img_shape = Img_test.shape[0]
                Img_test = scaler.transform(Img_test.reshape(img_shape, -1)).reshape(Img_test.shape)
                self.scaler = scaler

                print('Img_train shape : ', Img_train.shape)
                print('train samples : ', Img_train.shape[0])
                print('test samples : ', Img_test.shape[0])

                Lab_train = to_categorical(lab_train, classes)
                Lab_test = to_categorical(lab_test, classes)

                self.Img_train, self.Lab_train = Img_train, Lab_train
                self.Img_test, self.Lab_test = Img_test, Lab_test
                self.lab_train, self.lab_test = lab_train, lab_test

        def add_channels(self):

            img = self.img

            # NOTE : 흑백 이미지인지 검사
            if len(img.shape) == 3:
                width, height = img.shape[1:]


                # NOTE : theano 방식의 데이터 포맷을 사용하는 경우
                if K.image_dim_ordering() == 'th':
                    img = img.reshape(img.shape[0], 1, width, height)
                    in_shape = (1, width, height)

                # NOTE : tensorflow 방식의 데이터 포맷을 사용하는 경우
                else:
                    img = img.reshape(img.shape[0], width, height, 1)
                    in_shape = (width, height, 1)
            else:
                # NOTE : 컬러 이미지는 이미 채널 정보를 지니고 있음.
                in_shape = img.shape[1:]

            self.img = img
            self.in_shape = in_shape

    # NOTE : 모델 학습 및 성능평가 클래스
    class machine:
        def __init__(self, img, lab, classes = 2, fig = True):
            self.classes = classes
            self.fig = fig

            self.set_data(img, lab)
            self.set_model()


        def set_data(self, img, lab):
            classes = self.classes
            self.data = DATA(img, lab, classes)
            print('data.input_shape : ', self.data.in_shape)

        def set_model(self):
            classes = self.classes
            data = self.data
            self.model = CNN(classes = classes, in_shape = data.in_shape)

        # NOTE : verbose는 학습 진행상황 표시
        def fit(self, epo = 10, bat_size = 128, verbose = 1):
            data = self.data
            model = self.model

            early = EarlyStopping(monitor = 'val_loss', min_delta=0, patience= 30, verbose = 1, mode = 'auto')

            histo = model.fit(data.Img_train, data.Lab_train, batch_size = bat_size, callbacks = [early],
                              epochs = epo, validation_data=(data.Img_test, data.Lab_test))
            return histo

        def run(self, epo = 10, bat_size = 128, verbose = 1):
            data = self.data
            model = self.model
            fig = self.fig

            print('Confusion matrix')
            histo = self.fit(epo = epo, bat_size = bat_size, verbose = verbose)
            loss_n_accu = model.evaluate(data.Img_test, data.Lab_test, verbose = 0)

            test_pred = model.predict(data.Img_test, verbose = 0)
            pred = np.argmax(test_pred, axis = 1)
            print(metrics.confusion_matrix(data.lab_test, pred))

            print('Test Loss : ', loss_n_accu[0])
            print('Test Accuracy : ', loss_n_accu[1])

            # NOTE : 결과 저장
            suffix = ksp.unique_filename('datatime')
            dir_name = 'output_'+suffix

            os.chdir('./weights/CNN')
            snl = ksp.save_n_load('histo_his.npy', fold = dir_name)
            model.save_weights(suffix + 'dl_model.h5')
            print('Output results are saved in ', dir_name)

            if fig:
                plot = ksp.plot(histo)
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plot.plot_acc()

                plt.subplot(1,2,2)
                plot.plot_loss()

                plt.show()

            self.histo = histo
            return dir_name






