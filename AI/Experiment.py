'''
from os import getcwd, chdir

print('Present directory : ', getcwd())

chdir('..')
print('Changing directory : ', getcwd())


from keras.datasets import fashion_mnist
(_, _), (_, lab_test) = fashion_mnist.load_data()
print(len(set(lab_test)))

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
images = faces.images

array_to_img(images) '''

# from keras.utils.np_utils import to_categorical
'''
from PIL import Image
import numpy as np
from glob import glob
from matplotlib.pyplot import imshow
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import os
import matplotlib.pyplot as plt


fold_name = './Datas/cat_and_dog'
img_path = glob.glob(os.path.join(fold_name, '*.png'))
img = [Image.open(img_path[imgs]).convert('RGB') for imgs in range(len(img_path))]
labels = [img_path[rep][len(fold_name) + 1] for rep in range(len(img_path))]

w, h = img[0].size
s = min(w, h)
y = (h-s) // 2
x = (w-s) // 2

img = [img[rep].crop((x, y, s, s)) for rep in range(len(img))]

target_size = 224
img = [image.img_to_array(img[rep].resize((target_size, target_size), Image.ANTIALIAS)) for rep in range(len(img))]
img = np.asarray(img).astype(np.uint8)

print(img.shape)
'''

'''
path = './Datas/cat_and_dog/using_model'
img_path = glob(os.path.join(path, '*.png'))
from keras.applications.imagenet_utils import preprocess_input
labels = [int(img_path[label][len(path) + 1]) for label in range(len(img_path))]
images = [plt.imread(img_path[imgs]) for imgs in range(len(img_path))]
images = np.asarray(images)
print(images[0])
images = preprocess_input(images)
print(images[0])ce/AI_reboot/Keras/3minute_keras/Experiment.py
Using TensorFlow backend.



from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
path = './Datas/kaggle_cat_vs_dog/train'
img_path = glob(os.path.join(path, '*.jpg'))
cat_img = [imread(img_path[rep]) for rep in range(1500)]
cat_img = np.asarray(cat_img)
width, height, channel = cat_img[1].shape
print(width, height, channel)

images = np.asarray(cat_imgs)
print(images.shape)
print(preprocess_input(images))
'''
# os.chdir(path)
'''
width, height, channel = images[0].shape
images.reshape(images.shape[0], width, height, channel)
cat_lab = [0 for _ in range(1000)]
dog_lab = [1 for _ in range(1000)]
labels = np.concatenate([cat_lab, dog_lab])


print(cat_lab)
print(len(cat_lab))
images = np.concatenate([cat_imgs, dog_imgs])
images = np.asarray(images)
print(images.shape)


path = './Datas'
img_path = glob(os.path.join(path, '*.jpg'))

path2 = './Datas'
img_path2 = glob(os.path.join(path2, '*.jpg'))

print(img_path)
print(os.getcwd())
print(img_path2)
'''
from keras.applications import vgg19
model = vgg19.VGG19()
print(model.summary())
