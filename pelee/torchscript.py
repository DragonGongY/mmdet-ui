import torch
import torchvision

import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from peleenet import build_net
from data import BaseTransform, VOC_CLASSES
from utils.core import *
from utils.pycocotools.coco import COCO

parser = argparse.ArgumentParser(description='Pelee Testing')
parser.add_argument('-c', '--config', default='configs/Pelee_COCO.py')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='VOC or COCO dataset')
parser.add_argument('-m', '--trained_model', default="./weights/Pelee_COCO_size304_epoch40.pth",
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-t', '--thresh', default=0.2, type=float,
                    help='visidutation threshold')
parser.add_argument('--show', action='store_true',
                    help='Whether to display the images')
args = parser.parse_args()

print_info(' ----------------------------------------------------------------------\n'
           '|                       Pelee Demo Program                              |\n'
           ' ----------------------------------------------------------------------', ['yellow', 'bold'])

global cfg
cfg = Config.fromfile(args.config)
anchor_config = anchors(cfg.model)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
net = build_net('test', cfg.model.input_size, cfg.model)
checkpoint = torch.load("./weights/Pelee_COCO_size304_epoch40.pth")
# net.load_state_dict(checkpoint["state_dict"])
net.cpu()
net.eval()

example = torch.rand(1,3,304,304).cpu()
traced_module = torch.jit.trace(net, example)

traced_module.save("torch_scrit.pt")
