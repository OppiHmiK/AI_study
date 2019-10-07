from sklearn.datasets import fetch_olivetti_faces
import keras.backend as K
assert K.image_data_format() == 'channels_last'

import DCNN as dcnn

class Machine(dcnn.machine):

    def __init__(self, conv, nh, pd):
        faces = fetch_olivetti_faces()
        img_train, lab_train = faces.images, faces.target
        super().__init__(img_train, lab_train, classes = 40, Conv = conv, Nh = nh, Pd = pd)

def main(conv, nh, pd):
    m = Machine(conv, nh, pd)
    m.run(epo = 100, batch = 10)


conv = [64, 128, 32]
nh = [50, 100, 25]
pd = [0.1, 0.2, 0.05]
main(conv, nh, pd)