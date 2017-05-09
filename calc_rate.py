import sys, os
import re
import cPickle
from string import Template
import pprint
from collections import OrderedDict
import numpy as np


from eval_net import LayerAnalyzer

import cfg
caffe_path = cfg.CAFFE_PATH
print caffe_path
sys.path.insert(0, caffe_path)
import caffe

def generate_vgg16_arch(input_dim=3,
                        num_filters=[64,64,128,128,256,256,256,
                                     512,512,512,512,512,512,
                                     4096, 4096, 1000],
                        layer_names =  ['conv1_1', 'conv1_2',
                                        'conv2_1', 'conv2_2',
                                        'conv3_1', 'conv3_2', 'conv3_3',
                                        'conv4_1', 'conv4_2', 'conv4_3',
                                        'conv5_1', 'conv5_2', 'conv5_3',
                                        'fc6', 'fc7', 'fc8']):
    assert len(num_filters) == 16 and len(layer_names) == 16, '>>> Wrong VGG16 config'
    arch = OrderedDict()
    dims = [input_dim] + num_filters
    for i, name in enumerate(layer_names):
        arch[name] = (dims[i+1], dims[i]) # name -> (num_filter, channel)
    return arch

def read_arch_from_file(file_name):
    num_filters = open(file_name, 'r').readlines()
#    print num_filters
    num_filters = [int(s.strip().split(' ')[1]) for s in num_filters]
    return generate_vgg16_arch(input_dim=3, num_filters=num_filters)

def calc_size(arch, filter_size=3*3, pool5_size=7*7):
    assert len(arch) == 16, '>>> calc_size: Invalid VGG16 arch'
    size = 0
    for name in arch:
        fac = arch[name][0] * arch[name][1]
#        old_size = size
        if 'conv' in name:
            size += fac * filter_size
        elif name == 'fc6':
            size += fac * pool5_size
        else:
            size += fac * 1
#        print size-old_size
    return size

def calc_compression_rate(arch):
    print 'new size:', calc_size(arch) * 1. / 1e6
    return calc_size(generate_vgg16_arch()) / float(calc_size(arch))


if __name__ == '__main__':

    model_folder = sys.argv[1]
    print model_folder
    arch_file = os.path.join(model_folder, 'arch')
    arch = read_arch_from_file(arch_file)

    print 'rate:', calc_compression_rate(arch)
