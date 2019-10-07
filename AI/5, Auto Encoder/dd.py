from keras.utils import get_file
import numpy as np

images = np.load('full_numpy_bitmap_cat.npy', allow_pickle=True)
print(len(images))