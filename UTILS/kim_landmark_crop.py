# USAGE
# python kim_landmark_crop.py -d dataset -m model.dat -p inner

from imutils import face_utils
import random
import argparse
import imutils
from imutils import paths
import time
import dlib
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to dataset')
ap.add_argument('-m', '--model', required=True, help='path to facial landmark predictor')
ap.add_argument('-p', '--path', required = True,help = 'path to save data')
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(12)
random.shuffle(imagePaths)

print('[System] loading facial landmark predictor...')
predictor = dlib.shape_predictor(args['model'])

for imagePath in imagePaths:

    print(imagePath)
    forder = imagePath.split(os.path.sep)[-2]
    fileName = imagePath.split(os.path.sep)[-1]

    image = cv2.imread(imagePath)
    output = image.copy()
    image = imutils.resize(image, width=800, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rect = dlib.rectangle(left = 214, top = 47, right = 707, bottom = 540) # NOTE : for small products
    # rect = dlib.rectangle(left = 120, top = 30, right = 750, bottom = 580) # NOTE : for large products

    # use our custom dlib shape predictor to predict the location
    # of our landmark coordinates, then convert the prediction to
    # an easily parsable NumPy array
    startTime = time.time()
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    print(time.time()-startTime)
    outerList = []
    # loop over the (x, y)-coordinates from our dlib shape
    # predictor model draw them on the image
    for (sX, sY) in shape:
        cv2.circle(output, (sX, sY), 1, (0, 0, 255), -1)
        coordList = []
        coordList.append(sX); coordList.append(sY)
        outerList.append(coordList)

    startY, endY = outerList[9][1], outerList[0][1]
    startX, endX = outerList[14][0], outerList[5][0]

    cx, cy = int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2)
    print(f'top coordinate : {startY}, bottom coordinate : {endY}')
    print(f'left coordinate : {startX}, right coordinate : {endX}')
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 1)
    cv2.rectangle(output, (cx - 130, cy - 120), (cx + 130, cy + 130), (255, 0, 0), 1)

    crop = image[cy - 120: cy + 130, cx - 120 : cx + 130] # NOTE : inner coordinate
    # crop = image[startY: endY, startX: endX] # NOTE : outer coordinate

    if not os.path.isdir(args['path']):
        os.mkdir(args['path'])
    cv2.imwrite(f'{args["path"]}/{time.time()}.jpg', crop)

    # show the frame
    cv2.imshow("Frame", output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break