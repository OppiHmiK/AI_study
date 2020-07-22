# USAGE
# python kim_landmark.py -i dataset -n model03
# import the necessary packages
from efficientnet import load_model
import pickle
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from imutils import face_utils
import random
import argparse
import imutils
import numpy as np
from imutils import paths
import time
import dlib
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'path to input image')
ap.add_argument('-n', '--number', required = True, help = 'your model number')
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args["image"])))
random.seed(12)
random.shuffle(imagePaths)

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[System] loading facial landmark predictor...")
print('----형상 내경 모델 로딩----')
model1 = load_model('model/inner120.hdf5')
lb1 = pickle.loads(open('model/inner.pickle', 'rb').read())
graph1 = tf.get_default_graph()

print('----형상 외경 모델 로딩----')
model2 = load_model('model/outer120.hdf5')
lb2 = pickle.loads(open('model/outer.pickle', 'rb').read())
graph2 = tf.get_default_graph()

print('----랜드마크 모델 로딩----')        
predictor_small = dlib.shape_predictor('model/small.dat')
predictor_large = dlib.shape_predictor('model/large.dat')

for imagePath in imagePaths:

    print(imagePath)
    forder = imagePath.split(os.path.sep)[-2]
    fileName = imagePath.split(os.path.sep)[-1]

    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800, height=600)
    cv2.flip(image, 0)
    output = image.copy()        
    image_sh = output.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    modelNum = args['number']
    if (modelNum == 'model03') or (modelNum == 'model04'):
        rect = dlib.rectangle(left = 150, top = 35, right = 720, bottom = 560)
        shape = predictor_large(gray, rect)
        shape = face_utils.shape_to_np(shape)

    elif (modelNum == 'model01') or (modelNum == 'model02') or (modelNum == 'model05'):
        rect = dlib.rectangle(left = 214, top = 47, right = 707, bottom = 540)
        shape = predictor_small(gray, rect)
        shape = face_utils.shape_to_np(shape)

    # use our custom dlib shape predictor to predict the location
    # of our landmark coordinates, then convert the prediction to
    # an easily parsable NumPy array


    outerList = []
    # loop over the (x, y)-coordinates from our dlib shape
    # predictor model draw them on the image
    for (sX, sY) in shape:
        cv2.circle(image, (sX, sY), 1, (0, 0, 255), -1)
        coordList = []
        coordList.append(sX); coordList.append(sY)
        outerList.append(coordList)

    startY, endY = outerList[9][1], outerList[0][1]
    startX, endX = outerList[14][0], outerList[5][0]

    cx, cy = int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2)

    try:
        crop_ori = image_sh[startY:endY, startX:endX]    #### 외경 crop위치
        crop_sec = image_sh[cy - 130: cy + 130, cx - 130:cx + 130] #### 내경 crop위치
    except:
        crop_ori = image_sh[100:200, 100:200]
        crop_sec = image_sh[100:200, 100:200]          

    crop = cv2.resize(crop_ori, (100,100)) # 외경
    crop = crop.astype('float') / 255.0
    crop = img_to_array(crop)
    crop = np.expand_dims(crop, axis=0)
    
    with graph1.as_default():
        proba1 = model1.predict(crop)[0]
        
    idx1 = np.argmax(proba1)
    label1 = lb1.classes_[idx1]
    # amount1= int(proba1[idx1] * 100)
    
    crop2 = cv2.resize(crop_sec, (100, 100)) # 내경
    crop2 = crop2.astype('float') / 255.0
    crop2 = img_to_array(crop2)
    crop2 = np.expand_dims(crop2, axis=0)
    
    with graph2.as_default():
        proba2 = model2.predict(crop2)[0]

    idx2 = np.argmax(proba2)
    label2 = lb2.classes_[idx2]
    # amount2 = int(proba2[idx2] * 100)

    if label1 == '0': # 외경 검사
        cv2.rectangle(output, (startX, startY), (endX, endY), (0,255,0),5)
        
        if label2 != '0' :# 내경검사
            cv2.rectangle(output, (cx-130, cy-130), (cx+130, cy+130), (0,0,255),5)

        else:
            cv2.rectangle(output, (cx-130, cy-130), (cx+130, cy+130), (0,255, 0),5)

    elif label1 == '1' or label1 == '2':
        cv2.rectangle(output, (startX, startY), (endX, endY), (0,0,255),5)

    cv2.imshow("Frame", output)
    key = cv2.waitKey(0) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break