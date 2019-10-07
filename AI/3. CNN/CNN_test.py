from keras.datasets import cifar10
import keras.backend as K

assert K.image_data_format() == 'channels_last'
import col_img_CNN as ciC

class Machine(ciC.machine):
    def __init__(self):
        (img_train, lab_train), (img_test, lab_test) = cifar10.load_data()
        super().__init__(img_train, lab_train, classes = 10)

def main():
    m = Machine()
    m.run(epo = 1000, bat_size = 300, verbose = 1)

if __name__ == '__main__':
    main()