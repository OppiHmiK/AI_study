# USAGE
# python predict.py --image valeo

# import the necessary packages
from object_detection.utils import label_map_util
from easyocr import Reader
from imutils import paths
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import math as mt
import argparse
import imutils
import shutil
import random
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument('-l', '--lang', default = 'en')
args = vars(ap.parse_args())

pack_check = os.system('pip freeze | grep easyocr')
if int(pack_check) == 256:
	try:
		os.system('pip install easyocr')
		print('[INFO] easyocr installation complete.')

	except Exception as e:
		print('[ERR] easyocr installation failed... ;(')
	
else:
	print('[INFO] easyocr is already installed.')


imagePaths = sorted(list(paths.list_images(args["image"])))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sesssion = tf.Session(config=config)

#######################################################################################################
reader = Reader(args['lang'].split(','), gpu = True)

def calTheta(**kwargs):
	x, y = kwargs['ori']
	fx, fy = kwargs['fix']
	centerX, centerY = kwargs['center']

	l = mt.sqrt((fx - x)**2 + (fy - y)**2)
	r = mt.sqrt((centerX - x)**2 + (centerY - y)**2)

	print(f'centerX : {centerX} \ncenterY : {centerY}')
	print(f'x : {x} \n y : {y}')
	print(f'len : {l} \nradius : {r}')
	theta = mt.degrees(mt.asin(l / (2*r)))
	return 2*theta

# initialize a set of colors for our class labels
num_classes = 9
min_confidence = 0.45
COLORS = np.random.uniform(0, 255, size=(num_classes, 3))

# initialize the model
model = tf.Graph()
model_path = 'model/frozen_inference_graph.pb'
label_path = 'model/classes.pbtxt'

with model.as_default():
	# initialize the graph definition
	graphDef = tf.GraphDef()

	# load the graph from disk
	with tf.gfile.GFile(model_path, "rb") as f:
		serializedGraph = f.read()
		graphDef.ParseFromString(serializedGraph)
		tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(label_path)
categories = label_map_util.convert_label_map_to_categories(
	labelMap, max_num_classes=num_classes,
	use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

# create a session to perform inference
with model.as_default():
	sess = tf.Session(graph=model)

# grab a reference to the input image tensor and the boxes
# tensor
imageTensor = model.get_tensor_by_name("image_tensor:0")
boxesTensor = model.get_tensor_by_name("detection_boxes:0")

# for each bounding box we would like to know the score
# (i.e., probability) and class label
scoresTensor = model.get_tensor_by_name("detection_scores:0")
classesTensor = model.get_tensor_by_name("detection_classes:0")
numDetections = model.get_tensor_by_name("num_detections:0")

for imagePath in tqdm(imagePaths):
	
	forder = imagePath.split(os.path.sep)[-2]
	fileName = imagePath.split(os.path.sep)[-1]

	os.makedirs(f'imgs/output/{forder}', exist_ok=True)
	print(imagePath)

	# load the image from disk
	image = cv2.imread(imagePath)
	img = image.copy()
	output = image.copy()
	rot = image.copy()
	#output = imutils.resize(output, width=640
	
	(H, W) = image.shape[:2]
	# crop = output[:H//2, :W//2]

	# cv2.imwrite(f'crop/{forder}/{fileName}', crop)
	startTime = time.time()
	# prepare the image for detection
	#image = image[0:480, 80:560]
	(H, W) = image.shape[:2]
	image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
	image = np.expand_dims(image, axis=0)

	# perform inference and compute the bounding boxes,
	# probabilities, and class labels
	(boxes, scores, labels, N) = sess.run(
		[boxesTensor, scoresTensor, classesTensor, numDetections],
		feed_dict={imageTensor: image})

	# squeeze the lists into a single dimension
	boxes = np.squeeze(boxes)
	scores = np.squeeze(scores)
	labels = np.squeeze(labels)

	error = False
	idx = 0

	first_rotate = True
	# loop over the bounding box predictions
	for (box, score, label) in zip(boxes, scores, labels):
		# if the predicted probability is less than the minimum
		# confidence, ignore it
		if score < min_confidence:
			continue

		label = categoryIdx[label]
		idx = int(label["id"])
		showlabel = f'label : {label}'

		coord = coord_dict[label['name']]
		if idx == 9:
			# scale the bounding box from the range [0, 1] to [W, H]
			(startY, startX, endY, endX) = box
			startX = int(startX * W)
			startY = int(startY * H)
			endX = int(endX * W)
			endY = int(endY * H)


			# draw the prediction on the output image
			ori_pos = (endX - startX) // 2, (endY - startY) // 2

			coord = [380, 125, 760, 280] if 'ST' in imagePath else [380, 15, 760, 280]
			fix_pos = (W // 2, 165) if 'ST' in imagePath else (W // 2, 0)
			center = (W // 2, H // 2)

			print(f'fix_pos : {showlabel} \n{fix_pos}')
			if ((0 > startX or startX > (W // 2)) or (0 > startY or startY > (H // 2))) or ((0 > endX or endX > (W // 2)) or (0 > endY or endY > (H//2))):
				continue

			if first_rotate:
				os.makedirs(f'imgs/rotate/{forder}/{fileName.replace(".jpg", "")}', exist_ok=True)
				os.makedirs(f'imgs/crop/{forder}/{fileName.replace(".jpg", "")}', exist_ok=True)
				theta = calTheta(ori = ori_pos, fix = fix_pos, center = center)

				
				print('theta : ', theta)
				rot = imutils.rotate(rot, -1*theta)

				for idx in range(4):

					crop = rot[coord[1] : coord[1] + coord[3], coord[0]: coord[0] + coord[2]]
					cv2.imwrite(f'imgs/crop/{forder}/{fileName.replace(".jpg", "")}/{idx}_{fileName}', crop)

					rot = imutils.rotate(rot, -90)
					cv2.imwrite(f'imgs/rotate/{forder}/{fileName.replace(".jpg", "")}/{idx}_{fileName}', rot)

					result = reader.readtext(crop)
					for (_, text, _) in result:
						print(f'detected : ', text)

					print('\n')
				first_rotate = False

			
			cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(output, showlabel, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

			error = True
			idx += 1
			cv2.imwrite(f'imgs/output/{forder}/{idx}_{fileName}',output)

		else:
			error = True
			idx += 1
			cv2.imwrite(f'imgs/output/{forder}/{idx}_{fileName}',output)
			continue



	if error == True:
		key = cv2.waitKey(1) & 0xFF
	else:
		key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break
	elif key == ord("s"):
		shutil.copy(imagePath, 'img/{}'.format(fileName))
