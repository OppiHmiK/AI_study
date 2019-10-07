# -*- coding : utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import time as t
import os
import random as rand
from PIL import Image
from glob import glob
from keras.applications.imagenet_utils import preprocess_input

class plot:

    def __init__(self, histo):
        self.histo = histo

    def plot_acc(self):
        histo = self.histo
        plt.plot(histo.history['acc'])
        plt.plot(histo.history['val_acc'])
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Test'], loc=0)

    def plot_loss(self):
        histo = self.histo
        plt.plot(histo.history['loss'])
        plt.plot(histo.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend(['Train', 'Test'], loc=0)

class preprocessing:

    def __init__(self, format = 'jpg', idx = 20):
        self.format = format
        self.idx = idx

    # NOTE : 현재시간을 초 단위로 구해 새로운 이름을 만드는 함수
    def random_name(self):
        rand.seed(t.ctime())
        alpha = 'abcdefghijklmnopqrstuvwxyz0123456789'
        nums = [rep for rep in range(1, len(alpha))]

        rand_1 = rand.sample(nums, 15)
        fname = ''
        for alphas in rand_1:
            fname += alpha[alphas]
        return fname

    def crop(self, file_path = None):

        format, idx = self.format, self.idx
        if file_path == None:
            file_path = '../Datas/CNN/reuse_train'
        path = glob(os.path.join(file_path, '*.' + format))
        images = [Image.open(path[imgs]) for imgs in range(idx)]
        cropped_imgs = []
        for imgs in range(len(images)):
            width, height = images[imgs].size
            s = min(width, height)
            x = (width - s) // 1.5
            y = (height - s) // 1.5
            image = images[imgs].crop((x, y, s, s))
            cropped_imgs.append(image)

        return images

    def resize(self, image, size = 150):
        idx = self.idx
        resized_image = [image[imgs].resize((size, size), Image.ANTIALIAS) for imgs in range(idx)]
        return resized_image

    def image_save(self, image, save_name, save_path=None):
        idx, format = self.idx, self.format
        if save_path == None:
            save_path = '../Datas/CNN/reuse_train/train/resized'
        os.chdir(save_path)
        for saves in range(idx):
            image[saves].save('{}{}.{}'.format(save_name, saves+1, 'jpg'))

    def image_load(self, save = True, path=None):
        format, idx = self.format, self.idx

        if path == None:
            path = '../Datas/CNN/reuse_train/train/resized'
        image = glob(os.path.join(path, '*.' + format))

        if not save:
            image = [Image.open(image[imgs]) for imgs in range(idx)]

        else:
            image = [imread(image[imgs]) for imgs in range(idx)]
            image = np.asarray(image)
            image = preprocess_input(image)

        return image



