# -*- coding : utf-8 -*-

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import  Conv2D, MaxPooling2D
from keras.layers import Input, BatchNormalization

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import callbacks
from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt
from matplotlib.image import imread

from sklearn.model_selection import train_test_split
from glob import glob
import tensorflow as tf
import numpy as np
import os

print(os.getcwd())
import keras_sub_pack as ksp

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

class model(models.Model):

    def __init__(self, conv, node, drop, classes = 2, in_shape = None):
        super().__init__()
        self.conv, self.node, self.drop = conv, node, drop
        self.classes, self.in_shape = classes, in_shape

        self.build()
        i, t = self.i, self.t
        super().__init__(i, t)
        self.compile(loss = 'binary_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

    def build(self):

        conv, node, drop = self.conv, self.node, self.drop
        classes, in_shape = self.classes, self.in_shape

        i = Input(shape = in_shape)
        c = Conv2D(conv[0], activation = 'relu', padding = 'same', kernel_size=(3,3))(i)
        for convs in range(1, len(conv)):
            c = Conv2D(conv[convs], activation = 'relu', padding = 'same', kernel_size=(3,3))(c)
            c = MaxPooling2D(pool_size = (2,2), strides = 2)(c)

        c = BatchNormalization()(c)
        c = Dropout(drop[0])(c)
        f = Flatten()(c)

        h = Dense(node[0], activation = 'relu')(f)
        for nodes in range(1, len(node)):
            h = Dense(node[nodes], activation = 'relu')(h)
            h = BatchNormalization()(h)
            h = Dropout(drop[nodes])(h)
        h = BatchNormalization()(h)
        t = Dense(classes, activation = 'softmax')(h)

        self.i, self.t = i, t

class DATA:
    def __init__(self, path, test_size = 0.33, rand_stat = 42):

        self.path = path

        self.preprocessing()
        in_data, tar_data = self.in_data, self.tar_data

        in_train, in_test, tar_train, tar_test = train_test_split(in_data, tar_data, test_size = test_size, random_state=rand_stat)

        self.in_train, self.tar_train = in_train, tar_train
        self.in_test, self.tar_test = in_test, tar_test

    def load(self):

        path = self.path
        img_path = glob(os.path.join(path, '*.jpg'))

        cat_imgs = [imread(img_path[imgs]) for imgs in range(1500)]
        cat_labs = [0 for _ in range(1500)]

        dog_imgs = [imread(img_path[imgs]) for imgs in range(1500, 3000)]
        dog_labs = [1 for _ in range(1500)]

        cat_imgs = np.asarray(cat_imgs)
        dog_imgs = np.asarray(dog_imgs)
        
        images = np.concatenate([cat_imgs, dog_imgs])
        labels = np.concatenate([cat_labs, dog_labs])

        images = preprocess_input(images)
        in_shape = images.shape[1:]
        self.images, self.labels = images, labels
        self.in_shape = in_shape

    def preprocessing(self):
        self.load()
        in_data, tar_data = self.images, self.labels
        classes = len(set(tar_data))

        in_data = in_data.astype('float32')
        in_data = preprocess_input(in_data)
        tar_data = to_categorical(tar_data, classes)
        self.in_data, self.tar_data = in_data, tar_data
        self.classes = classes

class run:

    def __init__(self, conv, node, drop, epo = 30, path = None, batch = 100, fig = 1):
        data = DATA(path)
        m = model(conv, node, drop, data.classes, data.in_shape)


        reduce = callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=10, factor = 0.1)
        checkpoint = callbacks.ModelCheckpoint('../weights/kaggle_cat_n_dog.h5', monitor = 'val_acc',
                                               save_best_only=True)
        Early = callbacks.EarlyStopping(monitor = 'val_loss', mode = 'auto', patience=20, min_delta = 0)
        callback = [reduce, checkpoint, Early]

        gener = ImageDataGenerator(horizontal_flip=True, rotation_range=30)
        flow = gener.flow(data.in_train, data.tar_train, batch_size = batch)
        histo = m.fit_generator(flow, epochs=epo, steps_per_epoch= len(data.in_train) / 50,
                                callbacks=callback, validation_data=(data.in_test, data.tar_test))
        # histo = m.fit(data.in_train, data.tar_train, batch_size = batch, epochs = epo,
        #             validation_split=0.33)
        loss_n_accu = m.evaluate(data.in_test, data.tar_test, batch_size = batch)
        print(f'Tess loss : {loss_n_accu[0]} \n accuracy : {loss_n_accu[1]}')

        if fig:
            plt.figure(figsize=(12, 5))
            plot = ksp.plot(histo)

            plt.subplot(1,2,1)
            plot.plot_loss()

            plt.subplot(1,2,2)
            plot.plot_acc()
            plt.show()

class model_reuse:

    def __init__(self, file_path, idx, format = 'jpg', weights_path = None):

        if weights_path == None:
            weights_path = '../weights/kaggle_cat_n_dog/kaggle_cat_n_dog.h5'
        else:
            weights_path = weights_path

        custom_obj = {'model' : models.Model}
        self.model = models.load_model(weights_path, custom_objects=custom_obj)
        self.file_path = file_path
        self.wpath = weights_path
        self.ksp_p = ksp.preprocessing(format = format, idx = idx)

    def load(self, size = 150):
        file_path = self.file_path
        ksp_p = self.ksp_p
        image = ksp_p.crop(file_path = file_path)
        image = ksp_p.resize(image, size = size)
        ksp_p.image_save(image, save_name = 'train')
        # image = ksp_p.image_load()
        # self.image = image

    def train(self, image, label = None, epo = 15, batch = 20):
        model = self.model
        w_path = self.wpath

        early = callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=20, min_delta=0)
        reduce = callbacks.ReduceLROnPlateau(monitor='val_acc', mode='auto', patience=10, factor=0.1)
        check_point = callbacks.ModelCheckpoint(monitor='val_acc', mode='auto', save_best_only=True, filepath=w_path)
        callback_lis = [early, reduce, check_point]

        label = to_categorical(label)
        model.fit(image, label, epochs = epo, batch_size = batch, validation_split = 0.33, callbacks = callback_lis)
        loss_n_accu = model.evaluate(image, label, batch_size = 50)
        print(f'Test loss : {loss_n_accu[0]} accuracy : {loss_n_accu[1]}')
        print(f'Training wights are saved in : ../weights/kaggle_cat_n_dog as kaggle_cat_n_dog.h5')

    def ensemble(self, images, save = False):
        model = self.model


        '''
        images = ksp_p.image_load(save = save, path = file_path)
        images = ksp_p.resize(images, size = 150)
        ksp_p.image_save(images, 'test', save_path = save_path)
        images = ksp_p.image_load('../Datas/CNN/reuse_train/test/resized')
        '''

        pred1 = model.predict(images)
        pred2 = model.predict(images)
        pred3 = model.predict(images)
        pred4 = model.predict(images)
        pred5 = model.predict(images)

        ensemble = 0.2*(pred1 + pred2 + pred3 + pred4 + pred5)
        return ensemble

path = '../Datas/CNN/kaggle_cat_vs_dog/train/using_model'
'''
conv = [32, 64, 128, 64]
nh = [100, 50, 150, 200]
pd = [0.1, 0.05, 0.15, 0.2]
run(path = path, node = nh, conv =conv, drop=pd, epo = 100, batch = 50)
'''

cat_label = [0 for _ in range(1500)]
dog_label = [1 for _ in range(1500)]
cat_label.extend(dog_label)
img_path = glob(os.path.join('../Datas/CNN/kaggle_cat_vs_dog/train/used_data/using_model', '*.jpg'))
images = [imread(img_path[imgs]) for imgs in range(len(img_path))]
images = np.asarray(images)
images = preprocess_input(images)

m = model_reuse('../Datas/CNN/reuse_train/train', idx = 20, format = 'jpeg')
m.train(images, cat_label, epo = 100, batch = 100)
ensemble = m.ensemble(images)

pred_ind = [np.argmax(ensemble[pred]) for pred in range(len(ensemble))]
labels = ['cat', 'dog']
pred_lab = [labels[pred_ind[rep]] for rep in range(len(ensemble))]
print(pred_lab)
