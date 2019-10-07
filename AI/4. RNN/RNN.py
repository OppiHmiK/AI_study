# -*- coding : utf-8 -*-

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models

class RNN_LSTM(models.Model):

    def __init__(self, max_feat = 20000, maxlen = 80):

        i = layers.Input((maxlen,))

        # NOTE : Embedding 단어를 의미 벡터로 바꾸는 계층
        h = layers.Embedding(max_feat, 128)(i)
        h = layers.LSTM(128, dropout = 0.2, recurrent_dropout = 0.2)(h)
        o = layers.Dense(1, activation = 'sigmoid')(h)

        super().__init__(i, o)
        self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class DATA:

    def __init__(self, max_feat = 20000, maxlen = 80):
        (in_train, tar_train), (in_test, tar_test) = imdb.load_data(num_words=max_feat)
        # NOTE : 문장의 길이 통일
        in_train = sequence.pad_sequences(in_train, maxlen=maxlen)
        in_test = sequence.pad_sequences(in_test, maxlen = maxlen)

        self.in_train, self.tar_train = in_train, tar_train
        self.in_test, self. tar_test  = in_test, tar_test

class main:

    def __init__(self, epo = 3, batch = 32, max_feat = 20000, maxlen = 80):

        data = DATA(max_feat = max_feat, maxlen = maxlen)
        model = RNN_LSTM(max_feat = max_feat, maxlen = maxlen)
        model.fit(data.in_train, data.tar_train, epochs = epo, batch_size = batch, validation_split = 0.3)
        loss_n_acc = model.evaluate(data.in_test, data.tar_test, batch_size=batch)
        print(f'Test loss : {loss_n_acc[0]} \n accuracy : {loss_n_acc[1]}')


m = main()




