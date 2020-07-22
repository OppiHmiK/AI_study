from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from bs4 import BeautifulSoup as bs
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch, torchvision
from imutils import paths
from glob import glob
from time import time
import numpy as np
import detectron2
import itertools
import random
import json
import csv
import cv2
import os

ap = ArgumentParser()
ap.add_argument('-d', '--dataset', required = True)
ap.add_argument('-w', '--width', required = True, type = int, help = 'insert your image width')
ap.add_argument('-n', '--height', required = True, type = int, help = 'insert your image height')
ap.add_argument('-r', '--random', required = False, type = bool, default=False)
args = vars(ap.parse_args())

setup_logger()
dataRoot = args['dataset']
W, H = args['width'], args['height']

def get_label_list(label_path):
    lb = []

    with open(label_path) as labels:
        file = labels.readlines()
        for f in file:
            f = f.replace('\n', '')
            lb.append(f)
    
    return lb

lbPath = os.path.join(dataRoot, 'classes.txt')
labels = get_label_list(lbPath)

#* NOTE : registered dataset
def get_dataset_dicts():

    imgPath = sorted(list(paths.list_images(f'{dataRoot}/image')))
    xml = glob(os.path.join(dataRoot, '*.xml'))

    dataDict = []
    print('[PLEASE WAIT] XML file Parsing...')
    with open(xml[0], 'rb') as f:
        soup = bs(f, 'lxml')
        print('[SYSTEM] XML file parsing complete!')

    names = soup.select('images > image')
    print('[SYSTEM] data dictionary generating start')
    startTime = time()

    for name in names:
        NAME = name['file']
        jpg_name = os.path.join(dataRoot, NAME) if not dataRoot in NAME else NAME

        if not jpg_name in imgPath:
            print(f'[SYSTEM] Pass because no image {jpg_name}')
            continue
        

        record = {}
        record['file_name'] = jpg_name
        record['height'] = int(H)
        record['width'] = int(W)

        objs = []
        boxes = name.select('box')
        try:
            for box in boxes:
                # print(box.select('label')[0].text)
                # print(labels)
                obj = {
                    'bbox':[
                        float(box['left']),
                        float(box['top']),
                        float(int(box['left']) + int(box['width'])),
                        float(int(box['top']) + int(box['height']))
                    ], 'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id' : labels.index(box.select('label')[0].text),
                    'iscrowd': 0
                }
                
                objs.append(obj)
        except Exception as e:
            print(f'[ERROR] {e}')
            continue

        record['annotations'] = objs
        dataDict.append(record)

    print('[INFO] data dictionary has generated!')
    print(f'{time() - startTime:.2f}')
    return dataDict

DatasetCatalog.register(f'{dataRoot}/train', get_dataset_dicts)
MetadataCatalog.get(f'{dataRoot}/train').set(thing_classes = labels)
metadata = MetadataCatalog.get(f'{dataRoot}/train')

if args['random']:
    def cv2_imshow(image):
        plt.figure(figsize=(20,20))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        
    dataset_dicts = get_dataset_dicts()
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])

        boxes = d['annotations']
        for box in boxes:
            lb = labels[int(box['category_id'])]
            print(f'[INFO] Box detected : {box["category_id"]}. {lb}')
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1])

cfg = get_cfg()
cfg.merge_from_file('configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml')
# print(cfg)
cfg.DATASETS.TRAIN = (f'{dataRoot}/train',)
print(f'{dataRoot}/train')
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 150000    # 300 iterations seems good enough, but you can certainly train longer
cfg.SOLVER.STEPS = (100000, 150000)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.OUTPUT_DIR = 'weights'

os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume = False)
trainer.train()
