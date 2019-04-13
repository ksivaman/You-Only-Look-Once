from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

# parse the arguments 
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()

#the COCO dataset will support 80 classes
start = 0 
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()
num_classes = 80
classes = load_classes("data/coco.names")

#load and initialize the CNN
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)

model.net_info["height"] = args.reso

inp_dim = int(model.net_info["height"])
#input validation
assert (inp_dim % 32 == 0), "resolution should be a multiple of 32"
assert (inp_dim > 32), "resolution has to be greater than 32"

if CUDA:
    model.cuda()

#test mode
model.eval()

#load images
try:
    imlist = [osp.join(osp.realpath('.', images, img)) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("Incorrect input file name input, please try again..exiting\n")
    exit()

#create output folder if it doesn't already exist, else add to it. 
if not os.path.exists(args.det):
    os.makedirs(args.det)

loaded_ims = [cv2.imread(x) for x in imlist]



