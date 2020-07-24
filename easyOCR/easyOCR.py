
from time import time
import numpy as np
import easyocr
import cv2
import os

#* I test with just one image.
img = cv2.imread('test2.jpg')
output = img.copy()

#* available english, japanese, chinese, korean, and so on,,,
#* language code -> english : en, japanese : ja, chinese : ch_sim, korean : ko
#* and available using gpu (gpu = True)
reader = easyocr.Reader(['en'], gpu = True)

#* motor coordinates
coords = [[600, 150], [600, 360], [600, 595],  [600, 832], [600, 1060]]
for (idx, coord) in enumerate(coords):

    crop = output[coord[0]: coord[0]+230, coord[1]: coord[1]+145]
    h, w, c = crop.shape
    print(h, w, c)

    blank = np.zeros((h, w, c), np.uint8)
    blank[:, :] = (255, 255, 255)

    #* anticlockwise rotation through 90
    matrix = cv2.getRotationMatrix2D((w/2, h/2), 270, 1)
    matrix = cv2.warpAffine(crop, matrix, (w, h))

    #* image Shapening (need more experiment)
    #* this code will make the blurred image clear.
    kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
    clear = cv2.filter2D(matrix, -1, kernel)

    os.makedirs('output', exist_ok=True)
    cv2.imwrite(f'output/before{idx}.jpg', matrix)
    cv2.imwrite(f'output/after{idx}.jpg', clear)

    startTime = time()
    text = reader.readtext(clear, batch_size = 2)
    print(f'[Time] {time() - startTime}')

    print(text[1:])

    #* text represented like this (bounding box coordinate, predicted string, confidence level)
    #* [([[39, 81], [113, 81], [113, 97], [39, 97]], 'nnunlo"', 0.0026137721724808216), 
    # ([[33, 93], [116, 93], [116, 119], [33, 119]], '795PICL', 0.23890626430511475), 
    # ([[36, 112], [108, 112], [108, 136], [36, 136]], '901654', 0.5906641483306885)]

    try:
        matrix = cv2.rectangle(clear, (text[1][0][0][0], text[1][0][0][1]), (text[1][0][2][0], text[1][0][2][1]), (0, 0, 255), 1)
        matrix = cv2.rectangle(clear, (text[2][0][0][0], text[2][0][0][1]), (text[2][0][2][0], text[2][0][2][1]), (0, 0, 255), 1)

        blank = cv2.putText(blank, text[1][1], (text[1][0][0][0], text[1][0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        blank = cv2.putText(blank, text[2][1], (text[2][0][0][0], text[2][0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    except:
        continue

    glueImg = np.hstack([clear, blank])
    cv2.imwrite(f'output/glueImg{idx}.jpg', glueImg)

