import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa

# ia.seed(1)

def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin,ymin,xmax,ymax])
        # print(bndboxlist)
    childNode = root.find("object")
    if childNode:
        bndbox = root.find('object').find('bndbox')
    else:
        pass
    return bndboxlist
# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))

def change_xml_list_annotation(root, image_id, new_target,saveroot,id):

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))


def mkdir(path):

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
         # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

if __name__ == "__main__":

    IMG_DIR = "/home/yanglian/Documents/PrePiece_data/all_img"
    XML_DIR = "/home/yanglian/Documents/PrePiece_data/all_label"

    AUG_XML_DIR = "/home/yanglian/Documents/PrePiece_data/aug_label" # 存储增强后的XML文件夹路径
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "/home/yanglian/Documents/PrePiece_data/aug_img" # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 40 # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []


    # 影像增强
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((0.8, 1.2)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)), # iaa.GaussianBlur(0.5),
	#iaa.GaussianBlur(0.5),
        iaa.AverageBlur(k=(2, 7)),
        iaa.Sharpen(alpha=(0, 0.8), lightness=(0.75, 1.25))
        #iaa.PerspectiveTransform(scale=(0.08, 0.2)),
        #iaa.Affine(
        #    translate_px={"x": 10, "y": 10},
        #    scale=(0.8, 0.95),
        #    rotate=(-10, 10)
        #)  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    # seq = iaa.Sequential(
    #     [
    #         # apply the following augmenters to most images
    #         iaa.Fliplr(0.5),  # horizontally flip 50% of all images
    #         iaa.Flipud(0.2),  # vertically flip 20% of all images
    #         # crop images by -5% to 10% of their height/width
    #         # sometimes(iaa.CropAndPad(
    #         #     percent=(-0.05, 0.1),
    #         #     pad_mode=ia.ALL,
    #         #     pad_cval=(0, 255)
    #         # )),
    #         sometimes(iaa.Affine(
    #             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #             # scale images to 80-120% of their size, individually per axis
    #             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
    #             rotate=(-45, 45),  # rotate by -45 to +45 degrees
    #             shear=(-16, 16),  # shear by -16 to +16 degrees
    #             order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
    #             cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
    #             mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    #         )),
    #         # execute 0 to 5 of the following (less important) augmenters per image
    #         # don't execute all of them, as that would often be way too strong
    #         iaa.SomeOf((0, 5),
    #                    [
    #                        sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
    #                        # convert images into their superpixel representation
    #                        iaa.OneOf([
    #                            iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
    #                            iaa.AverageBlur(k=(2, 7)),
    #                            # blur image using local means with kernel sizes between 2 and 7
    #                            iaa.MedianBlur(k=(3, 11)),
    #                            # blur image using local medians with kernel sizes between 2 and 7
    #                        ]),
    #                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
    #                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
    #                        # search either for all edges or for directed edges,
    #                        # blend the result with the original image using a blobby mask
    #                        iaa.SimplexNoiseAlpha(iaa.OneOf([
    #                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
    #                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
    #                        ])),
    #                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    #                        # add gaussian noise to images
    #                        iaa.OneOf([
    #                            iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
    #                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    #                        ]),
    #                        iaa.Invert(0.05, per_channel=True),  # invert color channels
    #                        # iaa.Add((-10, 10), per_channel=0.5),
    #                        # change brightness of images (by -10 to 10 of original value)
    #                        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    #                        # either change the brightness of the whole image (sometimes
    #                        # per channel) or change the brightness of subareas
    #                        # iaa.OneOf([
    #                        #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
    #                        #     iaa.FrequencyNoiseAlpha(
    #                        #         exponent=(-4, 0),
    #                        #         first=iaa.Multiply((0.5, 1.5), per_channel=True),
    #                        #         second=iaa.ContrastNormalization((0.5, 2.0))
    #                        #     )
    #                        # ]),
    #                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
    #                        iaa.Grayscale(alpha=(0.0, 1.0)),
    #                        sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
    #                        # move pixels locally around (with random strengths)
    #                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
    #                        # sometimes move parts of the image around
    #                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
    #                    ],
    #                    random_order=True
    #                    )
    #     ],
    #     random_order=True
    # )


    for root, sub_folders, files in os.walk(XML_DIR):

        for name in files:

            bndbox = read_xml_annotation(XML_DIR, name)

            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

                # 读取图片
                if os.path.exists(os.path.join(IMG_DIR, name[:-4] + '.jpg')):
                    img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                else:
                    img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.bmp'))
                img = np.array(img)

                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                            int(bbs_aug.bounding_boxes[0].y1),
                                            int(bbs_aug.bounding_boxes[0].x2),
                                            int(bbs_aug.bounding_boxes[0].y2)])
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                image_auged = bbs.draw_on_image(image_aug, thickness=0)
                Image.fromarray(image_auged).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list,AUG_XML_DIR,epoch)
                print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                new_bndbox_list = []

