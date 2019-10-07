# -*- coding : utf-8 -*-
import keras_sub_pack as ksp

'''
cat_path = '../Datas/kaggle_cat_vs_dog/train/cat'
cat_images = ksp.crop(file_path = cat_path, format = 'jpg', image_range = 1500)
cat_images = ksp.resize(cat_images, 150)
ksp.image_save(cat_images, '../Datas/kaggle_cat_vs_dog/train/using_model',format = '.jpg', string = 'cat', ind = 1500)
'''

dog_path = '../Datas/kaggle_cat_vs_dog/train/dog'
dog_images = ksp.crop(file_path = dog_path, format = 'jpg', image_range = 1500)
dog_images = ksp.resize(dog_images, 150)
ksp.image_save(dog_images, '../Datas/kaggle_cat_vs_dog/train/using_model',format = '.jpg', string = 'dog', ind = 1500)