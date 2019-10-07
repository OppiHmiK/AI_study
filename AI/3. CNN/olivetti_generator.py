# -*- coding : utf-8 -*-
__author__ = 'KimHippo'

from keras.preprocessing.image import ImageDataGenerator
import time as t
import olivetti

class generator(olivetti.main):
    def __init__(self, conv, nh, pd, epo = 15, step_per_epo = 10, gen_param_dic = None):

        super().__init__(conv, nh, pd, epo)
        self.set_generator(step_per_epo=step_per_epo, gen_param_dic = gen_param_dic)

    def set_generator(self, step_per_epo = 10, gen_param_dic = None):
        if gen_param_dic is not None:
            self.gener = ImageDataGenerator(**gen_param_dic)
        else:
            self.gener = ImageDataGenerator()

        seed = t.ctime()
        self.gener.fit(self.data.in_train, seed = seed)
        self.step_per_epo = step_per_epo

    def fit(self, epo = 10, batch = 64, verbose = 1):

        model, data, gener = self.model, self.data, self.gener
        step_per_epo = self.step_per_epo

        flow = gener.flow(data.in_train, data.out_train, batch_size=batch)
        histo = model.fit_generator(flow, epochs = epo, steps_per_epoch=step_per_epo,
                                    validation_data=(data.in_test, data.out_test))

        return histo

if __name__ == '__main__':

    conv = [32, 128, 64],
    nodes = [100, 200, 150]
    dropout = [0.05, 0.2, 0.15]

    run = generator(conv, nodes, dropout)
    run.fit()

