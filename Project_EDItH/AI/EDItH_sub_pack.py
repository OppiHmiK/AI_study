import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image
from glob import glob
import numpy as np
import os

class plot:

    def __init__(self, histo, row=1, col=2):

        self.histo = histo
        self.row, self.col = row, col

        self.plot_acc()
        self.plot_loss()

        plt.show()

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

class preprocess:

    def __init__(self, paths, size = 250, is_sub = True, idx = 20, format='jpg'):

        self.paths, self.size = paths, size

        format = format.lower()
        self.format, self.idx = format, idx

        if is_sub:
            images = np.asarray([])
            for path, _, _ in os.walk(paths):
                image_dir = np.asarray(glob(os.path.join(path, '*.'+format)))
                images = np.concatenate([images, image_dir])
            data = [image.load_img(img) for img in images]

        else:
            images = glob(os.path.join(paths, '*.' + format))
            data = [image.load_img(img) for img in images]

        self.data = data

    def crop(self):

        data = self.data
        cropped_images = []

        for img in data:
            w, h = img.size
            s = min(w, h)
            x = w - s // 1.5
            y = h - s // 1.5
            imgs = img.crop((x, y, s, s))
            cropped_images.append(imgs)

        return cropped_images

    def resize(self, cropped = False, w = 800, h = 400):
        size = self.size


        if cropped:
            data = self.crop()
            resized = [image.img_to_array(data[img].resize((w, h), Image.ANTIALIAS)) / 255.0 for img in range(len(data))]

        else:
            data = self.data
            resized = [image.img_to_array(data[img].resize((w, h), Image.ANTIALIAS)) / 255.0 for img in range(len(data))]

        return resized

    def labeling(self, number_of = 100, label = 2):

        rep = 0
        lis = []
        while rep < label:
            data = np.asarray(rep for _ in range(number_of))
            total = np.concatenate([lis, data])
            rep += 1

        return total


if __name__ == '__main__':

    path  = './'
    pre = preprocess(paths = path, is_sub=False, format = 'jpg')
    img = pre.resize(w = 800, h = 400)

    from matplotlib import image
    image.imsave('룰루2.jpg', img)
