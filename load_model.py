#!/usr/bin/env python

import sys, os
caffe_path = os.path.join('../', 'python')
sys.path.insert(0, caffe_path)

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys
import cPickle
import pprint

from eval_net import LayerAnalyzer
from collections import OrderedDict
from trim_vgg16 import generate_vgg16_arch

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)

    test_prototxt = 'vgg16/64_64_128_128_256_256_256_512_512_512_512_512_512_4096_4096_1000/train_test.prototxt'

    if len(sys.argv) > 1:
        model_prototxt = sys.argv[1]
    else:
        model_prototxt = '../ilsvrc-vgg/VGG_ILSVRC_16_layers.caffemodel'

    net = caffe.Net(test_prototxt, model_prototxt, caffe.TEST)

    arch = generate_vgg16_arch()
    analyzers = OrderedDict()
    for layer in arch:
        if layer == 'fc8':
            continue
        analyzers[layer] = LayerAnalyzer(layer, arch[layer][0])
    print analyzers

    for i in range(5):
        res = net.forward()
        for layer in analyzers:
            analyzers[layer].eval_output(net)

