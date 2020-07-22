from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from imutils import paths
from time import time
import cv2
import os

ap = ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = 'path to your dataset')
ap.add_argument('-s', '--savepath', required = True)
args = vars(ap.parse_args())

savePath = args['savepath']
if not os.path.isdir(savePath):
    os.makedirs(savePath)

cfg = get_cfg()
cfg.merge_from_file('configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml')

cfg.OUTPUT_DIR = 'weights'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_0009999.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55 #* min_confidence
predictor = DefaultPredictor(cfg)

imgPath = list(sorted(paths.list_images(args['dataset'])))

for (idx, img) in enumerate(imgPath):
    startTime = time()
    image = cv2.imread(img)
    output = predictor(image)
    print(f'[INFO] time : {time() - startTime}')
    v = Visualizer(image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale = 1.2)
    v = v.draw_instance_predictions(output['instances'].to('cpu'))
    cv2.imwrite(f'{savePath}/output_{idx}.jpg', v.get_image()[:, :, ::-1])

