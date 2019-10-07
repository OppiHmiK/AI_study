# -*- coding : utf-8 -*-

from keras.preprocessing.image import load_img, img_to_array, save_img
from keras import backend as K
import numpy as np
from keras.applications.vgg19 import vgg19
from scipy.optimize import fmin_l_bfgs_b
from time import time
import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

class image_retouching:

    def __init__(self, target_path, style_path, img_height=400):
        image = load_img(path)
        width, height = image.size
        img_width = int((width * img_height) / height)
        self.img_width, self.img_height = img_width, img_height
        self.path = target_path

        target_image = K.constant(self.preprocessing(target_path))
        style_image = K.constant(self.preprocessing(style_path))
        generate_image = K.placeholder((1, img_width, img_height, 3))
        self.target_image, self.style_image, self.generate_image = target_image, style_image, generate_image

    def preprocessing(self, path):
        width, height = self.img_width, self.img_height
        image = load_img(path, target_size=(width, height))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = vgg19.preprocess_input(image)

        return image

    def deprocessing(self, image):
                                                                          

class losses:

    def __init__(self, target_path, style_path):
        self.data = image_retouching(target_path, style_path)
        self.target_image = data.target_image
        self.style_image = data.style_image
        self.generate_image = data.generate_image

    def content_loss(self):
        target_image, combine_image = self.target_image, self.generate_image
        self.c_loss = 0.5*K.sum(K.square(target_image - generate_image))

    def gram_matrix(self, image):
        if K.image_data_format() != 'channels_first':
            feat = K.batch_flatten(K.permute_dimensions(image, (2, 0, 1)))
            gram = K.dot(feat, K.transpose(feat))
        return gram

    def style_loss(self):
        style_image, generate_image = self.style_image, self.generate_image
        data = self.data
        S = self.gram_matrix(style_image)
        G = self.gram_matrix(generate_image)

        channel = 3
        size = data.img_height*data.img_width
        s_loss = K.sum(K.square(S - G)) / (4*(channel**2)*(size**2))
        self.s_loss = s_loss

    def total_loss(self, image):
        data = self.data
        a = image[:, :data.img_height - 1, :data.img_width-1, :] - [:, :1, :data.img_width - 1, :]
        b = image[:, :data.img_height - 1, :data.img_width-1, :] - [:, :data.img_height-1, :1, :]
        return K.sum(K.pow(a+b, 1.25))

