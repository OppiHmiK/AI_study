from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras import callbacks
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import EDItH_sub_pack as esp
import tensorflow as tf
import numpy as np
from keras import models
import os
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


class model(Model):

    def __init__(self, conv, nodes, drop, classes=2, shape=None):

        self.conv, self.nodes, self.drop = conv, nodes, drop
        self.classes, self.shape = classes, shape

        self.build_model()
        i, t = self.i, self.t

        super().__init__(i, t)
        self.compile(loss = 'binary_crossentropy', optimizer = 'adadelta', metrics=['accuracy'])

    def build_model(self):

        conv, nodes, drop = self.conv, self.nodes, self.drop
        classes, shape = self.classes, self.shape

        i = Input(shape=shape)
        c = Conv2D(conv[0], kernel_size=(3,3), padding='same')(i)
        c = MaxPooling2D((2, 2))(c)
        c = BatchNormalization()(c)

        for con in range(len(conv)):
            c = Conv2D(conv[con], kernel_size=(3,3), padding='same')(c)
            c = MaxPooling2D((2,2))(c)
        c = BatchNormalization()(c)
        c = Dropout(drop[0])(c)
        f = Flatten()(c)

        h = Dense(nodes[0], activation = 'relu')(f)
        for node in range(1, len(nodes)):
            h = Dense(nodes[node], activation = 'relu')(h)
            h = Dropout(drop[node])(h)
        h = BatchNormalization()(h)
        t = Dense(classes, activation='softmax')(h)

        self.i, self.t = i, t

class DATA:

    def __init__(self, path, test_size = 0.33, rand_stat = 0):
        self. path = path

        self.train()
        img, lab = self.img, self.lab
        train_img, test_img, train_lab, test_lab = train_test_split(img, lab, test_size=test_size, random_state=rand_stat)

        self.train_img, self.test_img = train_img, test_img
        self.train_lab, self.test_lab = train_lab, test_lab

    def train(self):

        path = self.path

        preprocess = esp.preprocess(paths = path, format='jpg')
        images = preprocess.resize(cropped=False)

        label1 = np.asarray([0 for _ in range(800)])
        label2 = np.asarray([1 for _ in range(800)])

        label = np.concatenate([label1, label2])

        images = np.asarray(images)
        in_shape = images.shape[1:]
        images = images.astype('float32')

        classes = len(set(label))
        label = to_categorical(label, classes)

        self.img, self.lab = images, label
        self.in_shape, self.classes = in_shape, classes


class run:

    def __init__(self, conv, nodes, drop, epo = 30, path = None, batch = 50, fig = 1):

        data = DATA(path)
        m = model(conv, nodes, drop, data.classes, shape = data.in_shape)
        print(m.summary())

        checkpoint = callbacks.ModelCheckpoint('./weights/human_faces_e.h5', monitor='val_acc', save_best_only=True)
        reduce = callbacks.ReduceLROnPlateau(factor=0.01, patience=10, monitor='val_acc')
        Early = callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=20, min_delta=0)
        callback = [checkpoint, Early, reduce]

        gener = ImageDataGenerator(horizontal_flip=True, rotation_range=30)
        flow = gener.flow(data.train_img, data.train_lab, batch_size = batch)
        histo = m.fit_generator(flow, epochs=epo, steps_per_epoch=len(data.train_img) / 50,
                                callbacks=callback, validation_data=(data.test_img, data.test_lab))

        # histo = m.fit(data.in_train, data.tar_train, batch_size = batch, epochs = epo,
        #             validation_split=0.33)

        loss_n_accu = m.evaluate(data.test_img, data.test_lab, batch_size = batch)
        print(f'Tess loss : {loss_n_accu[0]} \n accuracy : {loss_n_accu[1]}')

        if fig:
            plt.figure(figsize=(12, 5))
            plot = esp.plot(histo)

            plt.subplot(1,2,1)
            plot.plot_loss()

            plt.subplot(1,2,2)
            plot.plot_acc()
            plt.show()


path = './datas/image/train_2'
conv = [32, 64, 128, 64]
nodes = [100, 50, 150, 200]
drop = [0.1, 0.05, 0.15, 0.2]

m = run(conv, nodes, drop, epo = 100, path = path)











