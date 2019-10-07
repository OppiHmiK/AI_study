from glob import glob
import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

path = './Artificial-Intelligence/AI_reboot/Keras/Datas/RNN'
txt_path = glob(os.path.join(path, '*.txt'))

text = [open(txt, 'r', encoding='UTF-8', errors='ignore').read() for txt in txt_path]
text = '\n'.join(text)

text = text.replace('\r\n', ' ')
text = text.replace('\n', ' ')
text = text.replace('\ufeff', '')
text = text.replace('\uf8f7', '')

chars = list(sorted(set(text)))
char_2_idx = {ch: idx for idx, ch in enumerate(chars)}

from keras.layers import Dense, Input, LSTM, TimeDistributed
from keras.optimizers import RMSprop
from keras.models import Model


def char_rnn_model(num_chars, num_layers, nodes=512, drop=0.1):
    i = Input(shape=(None, num_chars), name='input')
    h = LSTM(nodes, return_sequences=True)(i)

    for rep in range(num_layers - 1):
        h = LSTM(nodes, return_sequences=True)(h)
    o = TimeDistributed(Dense(num_chars, name='dense', activation='softmax'))(h)
    model = Model(inputs = [i], outputs=[o])
    opti = RMSprop(lr=0.01)
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = char_rnn_model(len(char_2_idx), num_layers=2, nodes=640, drop=0)

import numpy as np
import random


def data_generator(all_text, num_chars, CHUNK, batch):
    x = np.zeros((batch, CHUNK, num_chars))
    y = np.zeros((batch, CHUNK, num_chars))

    while True:
        for row in range(batch):
            idx = random.randrange(len(all_text) - CHUNK - 1)
            chunk = np.zeros((CHUNK + 1, num_chars))

            for rep in range(CHUNK + 1):
                chunk[rep, char_2_idx[all_text[idx + rep]]] = 1
            x[row, :, :] = chunk[:CHUNK]
            y[row, :, :] = chunk[1:]
        yield x, y

chunk = 320

model.fit_generator(
    data_generator(text, len(chars), batch=256, CHUNK=3000), epochs = 10)