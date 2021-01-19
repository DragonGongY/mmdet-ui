#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import labelme
import shutil
import random

try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)

def labelme2voc(input_dir, output_dir, annotations_file, trainval_percent, train_percent):
    print(input_dir)
    print(output_dir)
    output_dir = osp.join(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, "VOC2007", "JPEGImages"))
    os.makedirs(osp.join(output_dir, "VOC2007", "Annotations"))
    os.makedirs(osp.join(output_dir, "VOC2007", "ImageSets"))
    os.makedirs(osp.join(output_dir, "VOC2007", "ImageSets", "Main"))

    # os.makedirs(osp.join(output_dir, "VOC2012", "JPEGImages"))
    # os.makedirs(osp.join(output_dir, "VOC2012", "Annotations"))
    # os.makedirs(osp.join(output_dir, "VOC2012", "ImageSets"))
    # os.makedirs(osp.join(output_dir, "VOC2012", "ImageSets", "Main"))
    print("Creating dataset:", output_dir)
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(osp.join(input_dir, annotations_file)).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(output_dir, "VOC2007", "JPEGImages", base + ".jpg")
        out_xml_file = osp.join(output_dir, "VOC2007", "Annotations", base + ".xml")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        maker = lxml.builder.ElementMaker()
        if img.shape.count == 3:
            xml = maker.annotation(
                maker.folder(),
                maker.filename(base + ".jpg"),
                maker.database(),  # e.g., The VOC2007 Database
                maker.annotation(),  # e.g., Pascal VOC2007
                maker.image(),  # e.g., flickr
                maker.size(
                    maker.height(str(img.shape[0])),
                    maker.width(str(img.shape[1])),
                    maker.depth(str(img.shape[2])),
                ),
                maker.segmented(),
            )
        else:
            xml = maker.annotation(
                maker.folder(),
                maker.filename(base + ".jpg"),
                maker.database(),  # e.g., The VOC2007 Database
                maker.annotation(),  # e.g., Pascal VOC2007
                maker.image(),  # e.g., flickr
                maker.size(
                    maker.height(str(img.shape[0])),
                    maker.width(str(img.shape[1])),
                ),
                maker.segmented(),
            )

        bboxes = []
        labels = []
        for shape in label_file.shapes:
            if shape["shape_type"] != "rectangle":
                print(
                    "Skipping shape: label={label}, "
                    "shape_type={shape_type}".format(**shape)
                )
                continue

            class_name = shape["label"]
            class_id = class_names.index(class_name)

            (xmin, ymin), (xmax, ymax) = shape["points"]
            # swap if min is larger than max.
            xmin, xmax = sorted([xmin, xmax])
            ymin, ymax = sorted([ymin, ymax])

            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(class_id)

            xml.append(
                maker.object(
                    maker.name(shape["label"]),
                    maker.pose(),
                    maker.truncated(),
                    maker.difficult(),
                    maker.bndbox(
                        maker.xmin(str(xmin)),
                        maker.ymin(str(ymin)),
                        maker.xmax(str(xmax)),
                        maker.ymax(str(ymax)),
                    ),
                )
            )
        with open(out_xml_file, "wb") as f:
            f.write(lxml.etree.tostring(xml, pretty_print=True))

    createTxt(trainval_percent, train_percent, output_dir)

def createTxt(trainval_percent, train_percent, xmlfilepath_base):
    xmlfilepath = xmlfilepath_base + "VOC2007/Annotations"
    # xmlfilepath = '/home/pengbo/mmdetection/data/VOCdevkit/Annotations'
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    datalist = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(datalist, tv)
    train = random.sample(trainval, tr)

    ftrainval_txt = xmlfilepath_base + "VOC2007/ImageSets/Main/trainval.txt"
    ftest_txt = xmlfilepath_base + "VOC2007/ImageSets/Main/test.txt"
    ftrain_txt = xmlfilepath_base + "VOC2007/ImageSets/Main/train.txt"
    fval_txt = xmlfilepath_base + "VOC2007/ImageSets/Main/val.txt"

    ftrainval = open(ftrainval_txt, 'w')
    ftest = open(ftest_txt, 'w')
    ftrain = open(ftrain_txt, 'w')
    fval = open(fval_txt, 'w')

    # ftrainval = open('/home/pengbo/mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
    # ftest = open('/home/pengbo/mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
    # ftrain = open('/home/pengbo/mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
    # fval = open('/home/pengbo/mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')

    for i in datalist:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == "__main__":
    pass
