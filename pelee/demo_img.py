#-*- coding:utf-8 -*-
"""
** File name	: train.py
** Author		: 龚司宇
** Date			: 2020-04-09
** Description	: 
**
** History		:
** 修改:		南志捷：将算法整理成应用算法组内部风格:2020.04.09
"""

# 导入XXX工具包
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# 导入opencv工具包
import cv2

# 导入本地工具包
from pelee.layers.functions import Detect
from pelee.peleenet import build_net
from pelee.data import BaseTransform, VOC_CLASSES
from pelee.utils.core import *



def draw_detection(im, bboxes, scores, cls_inds, fps, labels, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        cv2.rectangle(imgcv,
                      (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                      (255,0,0), 3)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (int(box[0]), int(box[1] - 7)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    return imgcv


# if __name__ == "__main__":
def verify_one_image(image_path, config, trained_model):
    """ 模型识别程序的主函数
    """
    # 设定基本配置参数
    parser = argparse.ArgumentParser(description='Pelee Testing')
    parser.add_argument('-c', '--config', default='configs/Pelee_COCO.py')
    parser.add_argument('-d', '--dataset', default='COCO',
                        help='VOC or COCO dataset')
    parser.add_argument('-m', '--trained_model', default="/home/pengbo/mmdetection/pelee/weights/COCO/Pelee_COCO_size304_epoch1215.pth",
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-t', '--thresh', default=0.28, type=float,
                        help='visidutation threshold')
    parser.add_argument('--show', action='store_true',
                        help='Whether to display the images')
    args = parser.parse_args()

    print_info(' ----------------------------------------------------------------------\n'
            '|                       Pelee Demo Program                              |\n'
            ' ----------------------------------------------------------------------', ['yellow', 'bold'])

    # 根据 --config 参数 导入模型配置文件
    global cfg
    cfg = Config.fromfile(config)
    # 根据config文件导入anchor参数
    anchor_config = anchors(cfg.model)
    print_info('The Anchor info: \n{}'.format(anchor_config))
    
    priorbox = PriorBox(anchor_config)
    
    net = build_net('test', cfg.model.input_size, cfg.model)
    init_net(net, cfg, trained_model)
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

    # file_path = os.getcwd() + "/data/coco/class_names.txt"
    file_path = os.getcwd() + "/model/Pelee_COCO/class_names.txt"
    # cats = [_.strip().split(',')[-1]
    #         for _ in open('data/coco_labels.txt', 'r').readlines()]
    cats = [_.strip() for _ in open(file_path, 'r').readlines()]
    label_config = {'VOC': VOC_CLASSES, 'COCO': tuple(cats)}
    labels = label_config[args.dataset]



    # fname = "./imgs/COCO/1_1.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    loop_start = time.time()
    w, h = image.shape[1], image.shape[0]
    img = _preprocess(image).unsqueeze(0)
    img = img.cuda()
    print(img.shape)
    scale = torch.Tensor([w, h, w, h]).cuda()
    out = net(img)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    allboxes = []
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        # soft_nms = cfg.test_cfg.soft_nms
        # min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=False)
        keep = keep[:cfg.test_cfg.keep_per_class]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist() + [j] for _ in c_dets])

    loop_time = time.time() - loop_start
    print("loop_time: ", loop_time)
    allboxes = np.array(allboxes)
    # print(allboxes)
    if allboxes is not None:
        boxes = allboxes[:, :4]
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]
        im2show = draw_detection(image, boxes, scores, cls_inds, -1, labels, args.thresh)

        # cv2.imshow('test', im2show)
        # cv2.waitKey()
    # else:
    #     cv2.imshow("image", image)
    #     cv2.waitKey()
        return im2show
