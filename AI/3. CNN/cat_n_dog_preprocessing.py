from keras.preprocessing import image
import numpy as np
from glob import glob
import os
from PIL import Image
import matplotlib.pyplot as plt

img_size = 100
path = '../Datas/cat_and_dog'
img_path = glob(os.path.join(path, '*.png'))
img_path_d = glob(os.path.join(path, '1_dogs500.png'))


labels = [img_path[label][len(path) + 1] for label in range(len(img_path))]
images = [Image.open(img_path[imgs]).convert('RGB') for imgs in range(len(img_path))]


image = Image.open(img_path_d[0]).convert('RGB')
width, height = image.size

width, height = images[0].size
s = min(width, height)
x = (width - s) // 2
y = (width - s) // 2


length = len(img_path)

image = image.crop((x, y, s, s))
image = image.resize((img_size, img_size), Image.ANTIALIAS)

images = [images[imgs].crop((x, y, s, s)) for imgs in range(length)]
images = [images[imgs].resize((img_size, img_size), Image.ANTIALIAS) for imgs in range(length)]

os.chdir('../Datas/cat_and_dog')
if not os.path.isdir('using_model'):
    os.mkdir('using_model')
os.chdir('using_model')

for cats in range(0, 500):
    images[cats].save(f'0_cats_{cats+1}.png')

for dogs in range(501, len(images)):
    images[dogs].save(f'1_dogs_{dogs+1}.png')

image.save('1_dogs_1001.png')



