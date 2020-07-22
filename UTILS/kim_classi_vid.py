# import the necessary packages
# USAGE : $python kim_cl_test.py -v kia.avi -o  objectDetection/kia -c classification/kia
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from keras.preprocessing.image import img_to_array
from efficientnet import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video")
ap.add_argument("-o", "--out", required=True, help = "path to  detection model and label")
ap.add_argument("-c", "--classi", required = True, help = "path to classification model and label")
args = vars(ap.parse_args())

class EfficientNetIMG:
    def __init__(self):
        modelPath = args['classi'] + '/epoch_15.hdf5'
        labelPath = args['classi'] + '/rekModel.pickle'
        self.model = load_model(modelPath)
        self.lb = pickle.loads(open(labelPath, 'rb').read())

    def inspection(self, img):

        img = cv2.resize(img, (70, 70))

        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # startTime = time.time()
        proba = self.model.predict(img)[0]
        idx = np.argmax(proba)
        label = self.lb.classes_[idx]
        # print(time.time() - startTime)

        # print('EfficientNet = {} / {}'.format(idx, proba[idx]))
        return label, proba[idx]

ENI = EfficientNetIMG()

stream = cv2.VideoCapture(args['video'])
if not os.path.isdir('output'):
    os.mkdir('output')

output_path = f'output/out_{args["out"]}.avi'
writer = None

while True:
    # grab the next frame
    (grabbed, Frame) = stream.read()
    (H, W) = Frame.shape[:2]

    output = Frame.copy()

    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break

    crop = Frame[265: 320, 397: 452]
    (label2, amount) = ENI.inspection(crop)

    if label2 != 'ok':
        cv2.rectangle(output, (397, 265), (452, 320), (0, 0, 255), 2)
	
    else:
        cv2.rectangle(output, (397, 265), (452, 320), (0, 255, 0), 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 20, (W, H), True)

    cv2.imshow("Frame", output)
    writer.write(output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

stream.release()
writer.release()