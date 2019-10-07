# -*- coding : utf-8 -*-
from keras import models
from keras.applications.imagenet_utils import preprocess_input
from glob import glob
from matplotlib import image
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
import os


def model_ensemble(test_images):

    weights_path = '../weights/kaggle_cat_n_dog/kaggle_cat_n_dog.h5'
    custom_obj = {'model' : models.Model}
    model = models.load_model(weights_path, custom_objects=custom_obj)

    pred1 = model.predict(test_images)
    pred2 = model.predict(test_images)
    pred3 = model.predict(test_images)
    pred4 = model.predict(test_images)
    pred5 = model.predict(test_images)

    ensemble = 0.2 * (pred1+pred2+pred3+pred4+pred5)
    return ensemble

'''
path = '../Datas/CNN/cat_and_dog/using_model'
img_path = glob(os.path.join(path, '*.jpg'))
images = [imread(img_path[imgs]) for imgs in range(35, 70)]
images = np.asarray(images)
image = preprocess_input(images)
cat_label = [0 for _ in range(488)]
dog_label = [1 for _ in range(500)]
exact_ind = np.concatenate([cat_label, dog_label])

test_path = '../Datas/CNN/ii'
img_path = glob(os.path.join(test_path, '*.jpg'))
length = len(img_path)

images = [Image.open(img_path[imgs]) for imgs in range(length)]
print(images[0])
images = [images[imgs].resize((150, 150), Image.ANTIALIAS) for imgs in range(length)]
print(images[1])
for imgs in range(len(images)):
    images[imgs].save(f'../Datas/CNN/test_resize_{imgs}.jpg')
'''

path = '../Datas/CNN'
img_path2 = glob(os.path.join(path, '*.jpg'))
images = [imread(img_path2[imgs]) for imgs in range(len(img_path2))]
images = np.asarray(images)
image = preprocess_input(images)
label = [0, 1]
label = to_categorical(label)
print(images.shape)

'''
for rep in range(len(img_path2)):
    plt.subplot(2, 1, rep + 1)
    plt.imshow(image[rep])
plt.subplots_adjust(hspace= 0.5)
plt.title('Images with normalization')
plt.show()

ensemble = model_ensemble(image)
pred_ind = [np.argmax(ensemble[preds]) for preds in range(len(image))]
print(pred_ind)

'''
wpath = '../weights/kaggle_cat_n_dog/kaggle_cat_n_dog.h5'
custom_obj = {'model' : models.Model}
