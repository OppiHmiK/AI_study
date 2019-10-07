# -*- coding : utf-8 -*-

from keras.utils.np_utils import to_categorical
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras import models, layers
import numpy as np

class fashion_DNN(models.Model):

    def __init__(self, Ni, Nh, Pd, No):

        self.Ni, self.Nh, self.Pd, self.No = Ni, Nh, Pd, No
        hidden_1 = layers.Dense(Nh[0])
        hidden_2 = layers.Dense(Nh[1])
        hidden_3 = layers.Dense(Nh[2])
        drop_1 = layers.Dropout(Pd[0])
        drop_2 = layers.Dropout(Pd[1])
        drop_3 = layers.Dropout(Pd[2])
        out = layers.Dense(No)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        i = layers.Input(shape = (Ni, ))
        h1 = relu(hidden_1(i))
        d1 = drop_1(h1)
        h2 = relu(hidden_2(d1))
        d2 = drop_2(h2)
        h3 = relu(hidden_3(d2))
        d3 = drop_3(h3)
        o = softmax(out(d3))

        super().__init__(i, o)
        self.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    def load_data(self):

        (in_train, out_train), (in_test, out_test) = fashion_mnist.load_data()

        out_train = to_categorical(out_train)
        out_test = to_categorical(out_test)

        L, W, H = in_train.shape
        in_train = in_train.reshape(-1, W*H)
        in_test = in_test.reshape(-1, W*H)

        in_train, in_test = in_train / 255.0, in_test / 255.0
        return (in_train, out_train), (in_test, out_test)

    def plot(self,histo):

        plt.subplot(1,2,1)
        plt.plot(histo.history['acc'])
        plt.plot(histo.history['val_acc'])
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Test'], loc = 0)

        plt.subplot(1,2,2)
        plt.plot(histo.history['loss'])
        plt.plot(histo.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Test'], loc = 0)
        plt.show()

    def main(self, epochs):

        model = fashion_DNN(self.Ni, self.Nh, self.Pd, self.No)
        (in_train, out_train), (in_test, out_test) = model.load_data()
        histo = model.fit(in_train, out_train, epochs = epochs, batch_size=1000, validation_split=0.3)
        perform_test = model.evaluate(in_test, out_test, batch_size = 1000)
        print('Test Loss and Accuracy : ', perform_test)
        model.plot(histo)

        return model.predict(in_test)

    def prediction(self, args):

        pred = self.main(1000)
        pred_class = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                      'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        (_, _), (in_test, out_test) = self.load_data()

        for rep in args:
            pred_ind = np.argmax(pred[rep])
            exact_ind = np.argmax(out_test[rep])

            print('Prediction label : ', pred_class[pred_ind])
            print('Exact label : ', pred_class[exact_ind])


if __name__ == '__main__':
    DNN = fashion_DNN(784, [50, 100, 70], [0.1, 0.5, 0.3], 10)
    np.random.seed(999)
    rand = np.random.randint(1, 10000, 10)
    DNN.prediction(rand)
