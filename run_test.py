#!/usr/bin/env python

import sys, os
caffe_path = '/home/vip2/hkust/caffe/python'
#caffe_path = os.path.join('..', 'caffe', 'python')
sys.path.insert(0, caffe_path)

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys
import argparse
import glob
import cPickle
import pprint

import eval_net

def forward_all_val(net, batch_size=50):
    num_batch = 50000 / batch_size
    acc1, acc5 = 0, 0
    for i in range(num_batch):
        res = net.forward()
        acc1 += res['accuracy@1'].sum()
        acc5 += res['accuracy@5'].sum()
        if i % 10 == 0:
            print i
    return {'accuracy@1': acc1 / num_batch, 'accuracy@5': acc5 / num_batch}


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'usage: test_prototxt, model, gpu'
        sys.exit()
    test_prototxt = sys.argv[1]
    model = sys.argv[2]
    gpu = int(sys.argv[3])

    caffe.set_mode_gpu()
    caffe.set_device(gpu)

    net = caffe.Net(test_prototxt, model, caffe.TEST)
    res = forward_all_val(net)

    print 'test accuracy (all val examples):'
    print res
