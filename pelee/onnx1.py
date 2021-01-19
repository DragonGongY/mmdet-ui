#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import torch
from layers.functions import Detect
from peleenet import build_net
from data import BaseTransform, VOC_CLASSES
from utils.core import *


parser = argparse.ArgumentParser(description='Pelee Testing')
parser.add_argument('-c', '--config', default='configs/Pelee_COCO.py')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='VOC or COCO dataset')
parser.add_argument('-m', '--trained_model', default="./weights/Pelee_COCO_size304_epoch40.pth",
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-t', '--thresh', default=0.25, type=float,
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
init_net(net, cfg, args.trained_model)
print_info('===> Finished constructing and loading model', ['yellow', 'bold'])
net.eval()
num_classes = cfg.model.num_classes

with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

_preprocess = BaseTransform(
    cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(num_classes,
                  cfg.loss.bkg_label, anchor_config)

base = int(np.ceil(pow(num_classes, 1. / 3)))

cats = [_.strip().split(',')[-1]
        for _ in open('data/coco_labels.txt', 'r').readlines()]
label_config = {'VOC': VOC_CLASSES, 'COCO': tuple(['__background__'] + cats)}
labels = label_config[args.dataset]

def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        # box = [int(_) for _ in box]
        cv2.rectangle(imgcv,
                      (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                      (255,0,0), 3)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (int(box[0]), int(box[1] - 7)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    return imgcv
img = torch.randn(1,3,304,304)
img = img.cuda()
print(img.shape)
out = net(img)
torch.onnx.export(net, img, "./pelee.onnx", verbose=1)
