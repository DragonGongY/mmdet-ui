#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import numpy as np
import PIL.Image

import labelme
import random
import shutil

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def labelme2coco(input_dir, output_dir, annotations_file, trainval_percent, train_percent):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', default=input_dir, help='input annotated directory')
    parser.add_argument('--output_dir', default=output_dir, help='output dataset directory')
    parser.add_argument('--labels', default=annotations_file, help='labels file', required=False)
    args = parser.parse_args()

    if  osp.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'train'))
    os.makedirs(osp.join(args.output_dir, 'val'))
    os.makedirs(osp.join(args.output_dir, 'test'))
    print('Creating dataset:', args.output_dir)
    # print('Output directory already exists:', args.output_dir)
        # sys.exit(1)


    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))
        class_names.append(class_name)

    # save class_names.txt
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    out_annotations = osp.join(args.output_dir, 'annotations')
    if not os.path.exists(out_annotations):
        os.makedirs(out_annotations)


    label_files = glob.glob(osp.join(args.input_dir, '*.json'))

    num_total = len(label_files)
    num_trainval = int(num_total * trainval_percent)
    num_train = int(num_trainval * train_percent)
    trainval = random.sample(range(num_total), num_trainval)
    train = random.sample(trainval, num_train)
    train_files = []
    val_files = []
    test_files = []
    for i in range(num_total):
        if i in trainval:
            if i in train:
                train_files.append(label_files[i])
            else:
                val_files.append(label_files[i])
        else:
            test_files.append(label_files[i])

    for i in range(3):
        temp_file = []
        dir_name = None
        if i == 0:
            out_ann_file = osp.join(args.output_dir, "annotations", 'train.json')
            temp_file = train_files
            dir_name = "train"

            for image_id, label_file in enumerate(temp_file):
                print('Generating dataset from:', label_file)
                with open(label_file) as f:
                    label_data = json.load(f)

                base = osp.splitext(osp.basename(label_file))[0]
                out_img_file = osp.join(
                    args.output_dir, dir_name, base + '.jpg'
                )

                img_file = osp.join(
                    osp.dirname(label_file), label_data['imagePath']
                )
                img = np.asarray(PIL.Image.open(img_file).convert('RGB'))
                PIL.Image.fromarray(img).save(out_img_file)
                data['images'].append(dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                ))

                masks = {}  # for area
                segmentations = collections.defaultdict(list)  # for segmentation
                for shape in label_data['shapes']:
                    points = shape['points']
                    label = shape['label']
                    group_id = shape.get('group_id')
                    shape_type = shape.get('shape_type')
                    mask = labelme.utils.shape_to_mask(
                        img.shape[:2], points, shape_type
                    )

                    if group_id is None:
                        group_id = uuid.uuid1()

                    instance = (label, group_id)

                    if instance in masks:
                        masks[instance] = masks[instance] | mask
                    else:
                        masks[instance] = mask

                    points = np.asarray(points).flatten().tolist()
                    segmentations[instance].append(points)
                segmentations = dict(segmentations)

                for instance, mask in masks.items():
                    cls_name, group_id = instance
                    if cls_name not in class_name_to_id:
                        continue
                    cls_id = class_name_to_id[cls_name]

                    mask = np.asfortranarray(mask.astype(np.uint8))
                    mask = pycocotools.mask.encode(mask)
                    area = float(pycocotools.mask.area(mask))
                    bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                    data['annotations'].append(dict(
                        id=len(data['annotations']),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    ))

            with open(out_ann_file, 'w') as f:
                json.dump(data, f)

        elif i == 1:
            out_ann_file = osp.join(args.output_dir, "annotations", 'val.json')
            temp_file = val_files
            dir_name = "val"

            for image_id, label_file in enumerate(temp_file):
                print('Generating dataset from:', label_file)
                with open(label_file) as f:
                    label_data = json.load(f)

                base = osp.splitext(osp.basename(label_file))[0]
                out_img_file = osp.join(
                    args.output_dir, dir_name, base + '.jpg'
                )

                img_file = osp.join(
                    osp.dirname(label_file), label_data['imagePath']
                )
                img = np.asarray(PIL.Image.open(img_file).convert('RGB'))
                PIL.Image.fromarray(img).save(out_img_file)
                data['images'].append(dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                ))

                masks = {}  # for area
                segmentations = collections.defaultdict(list)  # for segmentation
                for shape in label_data['shapes']:
                    points = shape['points']
                    label = shape['label']
                    group_id = shape.get('group_id')
                    shape_type = shape.get('shape_type')
                    mask = labelme.utils.shape_to_mask(
                        img.shape[:2], points, shape_type
                    )

                    if group_id is None:
                        group_id = uuid.uuid1()

                    instance = (label, group_id)

                    if instance in masks:
                        masks[instance] = masks[instance] | mask
                    else:
                        masks[instance] = mask

                    points = np.asarray(points).flatten().tolist()
                    segmentations[instance].append(points)
                segmentations = dict(segmentations)

                for instance, mask in masks.items():
                    cls_name, group_id = instance
                    if cls_name not in class_name_to_id:
                        continue
                    cls_id = class_name_to_id[cls_name]

                    mask = np.asfortranarray(mask.astype(np.uint8))
                    mask = pycocotools.mask.encode(mask)
                    area = float(pycocotools.mask.area(mask))
                    bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                    data['annotations'].append(dict(
                        id=len(data['annotations']),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    ))

            with open(out_ann_file, 'w') as f:
                json.dump(data, f)

        elif i == 2:
            out_ann_file = osp.join(args.output_dir, "annotations", 'test.json')
            temp_file = test_files
            dir_name = "test"

            for image_id, label_file in enumerate(temp_file):
                print('Generating dataset from:', label_file)
                with open(label_file) as f:
                    label_data = json.load(f)

                base = osp.splitext(osp.basename(label_file))[0]
                out_img_file = osp.join(
                    args.output_dir, dir_name, base + '.jpg'
                )

                img_file = osp.join(
                    osp.dirname(label_file), label_data['imagePath']
                )
                img = np.asarray(PIL.Image.open(img_file).convert('RGB'))
                PIL.Image.fromarray(img).save(out_img_file)
                data['images'].append(dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                ))

                masks = {}  # for area
                segmentations = collections.defaultdict(list)  # for segmentation
                for shape in label_data['shapes']:
                    points = shape['points']
                    label = shape['label']
                    group_id = shape.get('group_id')
                    shape_type = shape.get('shape_type')
                    mask = labelme.utils.shape_to_mask(
                        img.shape[:2], points, shape_type
                    )

                    if group_id is None:
                        group_id = uuid.uuid1()

                    instance = (label, group_id)

                    if instance in masks:
                        masks[instance] = masks[instance] | mask
                    else:
                        masks[instance] = mask

                    points = np.asarray(points).flatten().tolist()
                    segmentations[instance].append(points)
                segmentations = dict(segmentations)

                for instance, mask in masks.items():
                    cls_name, group_id = instance
                    if cls_name not in class_name_to_id:
                        continue
                    cls_id = class_name_to_id[cls_name]

                    mask = np.asfortranarray(mask.astype(np.uint8))
                    mask = pycocotools.mask.encode(mask)
                    area = float(pycocotools.mask.area(mask))
                    bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                    data['annotations'].append(dict(
                        id=len(data['annotations']),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    ))

            with open(out_ann_file, 'w') as f:
                json.dump(data, f)

if __name__ == '__main__':
    pass
