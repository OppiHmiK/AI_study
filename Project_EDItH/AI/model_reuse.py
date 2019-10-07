# -*- coding : utf-8 -*-

from keras import models
from glob import glob
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
import os
import EDItH_sub_pack as esp



class model_load:

    def __init__(self, fpath, fname = '/human_faces.h5', weight_path = None):

        if weight_path == None:
            weight_path = './weights'

        custom_obj = {'model' : models.Model}
        m = models.load_model(weight_path + fname, custom_objects=custom_obj)

        prepo = esp.preprocess(fpath, is_sub = False)
        resized = np.asarray(prepo.resize(cropped=False))

        image = resized.astype('float32')
        image /= 255.0

        self.m, self.img = m, image

    def ensemble(self):

        m, img = self.m, self.img

        p1 = m.predict(img)
        p2 = m.predict(img)
        p3 = m.predict(img)
        p4 = m.predict(img)

        return 0.25 *(p1+p2+p3+p4)

model = model_load(fpath = './datas')
print(np.argmax(model.ensemble()[1]))
