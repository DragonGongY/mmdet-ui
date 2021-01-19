# -*- coding: utf-8 -*-
import sys

import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision
import os.path as osp
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.clear_session()
K.set_image_dim_ordering('tf')

import test
import tensorflow as tf

import torch
from torch import nn
from torchsummary import summary
from torch.autograd import Variable
import tensorflow
from tensorflow.python.keras.backend import get_session
from tensorflow.python.keras.models import load_model
from tensorflow.python.framework import graph_util, graph_io

from keras.utils import plot_model
# K.set_image_data_format('channels_first') 0
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def check_error(output, k_model, input_np, epsilon=1e-3):
    pytorch_output = output[0].data.cpu().numpy()
    # pytorch_output = np.max(pytorch_output)
    # print('torch:',pytorch_output)
    # print('=====================')
    # print('torch:',pytorch_output)
    keras_output = k_model.predict(input_np)
    keras_output = keras_output[0]
    # keras_output = np.max(keras_output)
    # print('=====================')
    # print('keras pre:',keras_output)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


import numpy as np


def normalization0_1(data):
    _range = np.max(data) - np.min(data)
    data = (data - np.min(data)) / _range
    mean = [0.485, 0.456, 0.406]
    std_ad = [0.229, 0.224, 0.225]
    return np.divide(np.subtract(data, mean), std_ad)


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", ):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = ["output_0_1"]  ##get from init_graph
    # out_nodes.append(out_prefix + str(0))
    tf.identity(h5_model.output[0], out_prefix + str(0))
    sess = get_session()
    init_graph = sess.graph.as_graph_def()  ##get out_nodes
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)


if __name__ == '__main__':
    ##step1: load pytorch model
    # model = test.main()
    model = torch.load("/home/dp/Desktop/algorithms/Pelee.Pytorch/weights/Pelee_COCO_size304_epoch40.pth")
    model = model.cuda()  ##cuda
    summary(model, (3, 304, 304))  ##summary(model, (channels, pic_h, pic_w))
    model.eval()

    ##step2: pytorch .pth to keras .h5  and test .h5
    input_np = np.random.uniform(0, 1, (1, 3, 304, 304))
    input_var = Variable(torch.FloatTensor(input_np)).cuda()  ##cuda
    # input_var = Variable(torch.FloatTensor(input_np))
    k_model = pytorch_to_keras(model, input_var, (3, 304, 304,), verbose=True, name_policy='short')
    k_model.summary()
    k_model.save('my_model.h5')

    output = model(input_var)
    check_error(output, k_model, input_np)  ## check the error between .pth and .h5

    ##step3: load .h5 and .h5 to .pb
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)  ##不可少，
    my_model = load_model('my_model.h5')
    h5_to_pb(my_model, output_dir='./model/', model_name='model.pb')

    ##step4:  load .pb and test .pb
    pb_path = './model/model.pb'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(pb_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")

        pic_file = './datasets/data'
        pic_list = os.listdir(pic_file)
        for name in pic_list:
            img_path = '{}/{}'.format(pic_file, name)
            im = cv2.imread(img_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img = cv2.resize(im, (304, 304))
            img = np.asarray(img, dtype=np.float32)
            img = normalization0_1(img)
            img_data = np.transpose(img, (2, 0, 1))
            img_input = np.asarray(img_data, dtype=np.float32)[np.newaxis, :, :, :]

            input = sess.graph.get_tensor_by_name("input_0:0")
            output = sess.graph.get_tensor_by_name("output_0_1:0")
            pre_label = sess.run([output], feed_dict={input: img_input})
            pre_label = pre_label[0][0]
            # print(pre_label)
            pre_label = np.argmax(softmax(pre_label))
            print('------------------------')
            print('{} prelabel is {}'.format(pic_name, pre_label))
