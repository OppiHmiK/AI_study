from pyimagesearch.utils.tfannotation import TFAnnotation
from argparse import ArgumentParser
from imutils import paths
import tensorflow as tf
from PIL import Image
import json
import os

ap = ArgumentParser()
ap.add_argument('-d', '--dataset', required = True)
ap.add_argument('-s', '--savepath', required = True)
ap.add_argument('-p', '--percentage', required = False, type = float, default = 0.25)
opt = ap.parse_args()

outputPath = opt.savepath
jsonPath = sorted(paths.list_files(f'{opt.dataset}'))

trainPath = jsonPath[int(len(jsonPath) * opt.percentage): ]
testPath = jsonPath[ :int(len(jsonPath) * opt.percentage)]

datasets = [('train', trainPath, outputPath + '/training.record'), ('test', testPath, outputPath + '/testing.record')]
os.makedirs(outputPath, exist_ok = True)
CLASSES_TEXT = f'{outputPath}/classes.pbtxt'

CLASSES = {}

def main():

    lbList = []
    for (dtype, jsonPath, savePath) in datasets:

        writer = tf.python_io.TFRecordWriter(savePath)
        total = 0

        for jsonFile in jsonPath:
            # print(jsonFile)

            with open(jsonFile) as f:
                annotData = json.load(f)

            for info in annotData['shapes']:
                imagePath = annotData['imagePath'].replace('../', './')

                lb = info['label']
                encoded = tf.gfile.GFile(imagePath, 'rb').read()
                encoded = bytes(encoded)

                pilImg = Image.open(imagePath)
                (W, H) = pilImg.size[:2]

                fileName = imagePath.split(os.path.sep)[-1]
                encoding = fileName[fileName.rfind('.') + 1: ] #* jpg

                tfAnnot = TFAnnotation()
                tfAnnot.image = encoded
                tfAnnot.encoding = encoding
                tfAnnot.filename = fileName
                tfAnnot.width = W
                tfAnnot.height = H
                
                box = info['points']
                startX, startY = max(0, float(box[0][0])), max(0, float(box[0][1]))
                endX, endY = min(W, float(box[1][0])), min(H, float(box[1][1]))

                xMin = startX / W
                xMax = endX / W

                yMin = startY / H
                yMax = endY / H

                if xMin > xMax or yMin > yMax:
                    continue

                elif xMax < xMin or yMax < yMin:
                    continue

                lbList.append(lb)
                lbList = sorted(list(set(lbList)))


                for (idx, lb) in enumerate(lbList):
                    CLASSES[lb] = idx + 1
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(lb.encode('utf8'))
                tfAnnot.classes.append(CLASSES[lb])
                tfAnnot.difficult.append(0)

                print(CLASSES)

            total += 1
            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)

            # add the example to the writer
            writer.write(example.SerializeToString())

        writer.close()
        print("[INFO] {} examples saved for '{}'".format(total, dtype))
        print(lbList)
    
    f = open(CLASSES_TEXT, "w")
    # loop over the classes
    for (k, v) in CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)

    # close the output classes file
    f.close()


if __name__ == '__main__':
    main()