import sys, os
import re
import cPickle
from string import Template
import pprint
from collections import OrderedDict
import numpy as np


from eval_net import LayerAnalyzer

import cfg
caffe_path = '/home/vip2/hkust/caffe/python'
print caffe_path
sys.path.insert(0, caffe_path)
import caffe

def load_model(ptx, model, device_id=1, mode=caffe.TEST):
    caffe.set_mode_gpu()
    caffe.set_device(device_id)
    if model is None:
        net = caffe.Net(ptx, mode)
    else:
        net = caffe.Net(ptx, model, mode)
    print 'Finish loading: {0}\n\tprototxt: {1}'.format(model, ptx)
    return net

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
    return calc_size(generate_vgg16_arch()) / float(calc_size(arch))

def generate_name(arch):
    num_filters = [str(l[0]) for l in arch.values()]
    name = '_'.join(num_filters)
    return name

def create_proto(arch, path):
    assert os.path.exists(path), '>>> Path {0} does not exist'.format(path)
    train_test_tpl = 'template/train_val.prototxt.tpl'

    name_to_nfilters = {}
    for name in arch:
        name_to_nfilters[name] = arch[name][0]
    with open(train_test_tpl, 'r') as f:
        tpl_str = f.read()
        t = Template(tpl_str)
        s = t.substitute(name_to_nfilters)

    train_test_ptx = os.path.join(path, cfg.TRAIN_TEST_PTX)
    with open(train_test_ptx, 'w') as f:
        f.write(s)
    print 'writing', train_test_ptx

    solver_tpl = 'template/solver.prototxt.tpl'
#    snapshot_pfx = os.path.join(path, 'vgg16')
    with open(solver_tpl, 'r') as f:
        tpl_str = f.read()
        t = Template(tpl_str)
        s = t.substitute(train_proto=cfg.TRAIN_TEST_PTX, snapshot_prefix='vgg16')

    solver_ptx = os.path.join(path, cfg.SOLVER_PTX)
    with open(solver_ptx, 'w') as f:
        f.write(s)
    print 'writing', solver_ptx
    return train_test_ptx, solver_ptx

class VGG16Model:
    def __init__(self, arch, loc):
        self.arch = OrderedDict(arch) # copy?
        # self.name = generate_name(arch)
        self.loc = loc
        # os.path.join(os.getcwd(), path, self.name)
        self.log_file = os.path.join(self.loc, 'log.txt')
        self.compression_rate = calc_compression_rate(arch)

#        self.train_test_ptx, self.solver_ptx = create_proto(arch, self.loc)

        if not os.path.exists(self.loc):
            os.makedirs(self.loc)
            self.train_test_ptx, self.solver_ptx = create_proto(arch, self.loc)
        else:
            self.train_test_ptx = os.path.join(self.loc, cfg.TRAIN_TEST_PTX)
            self.solver_ptx = os.path.join(self.loc, cfg.SOLVER_PTX)

    def load_net(self, weight_init=None):
        pass

    def _select_model(self):
        pass

def trim_filters(arch, analyzers):
    assert len(arch) == 16
    new_arch = OrderedDict()
    filter_maps = {}
    channel_maps = {}
    for layer in arch:
        new_arch[layer] = list(arch[layer])
        filter_maps[layer] = range(arch[layer][0])
        channel_maps[layer] = range(arch[layer][1])

    layers = arch.keys()
    for layer in analyzers:
        pos = layers.index(layer)
        assert pos < len(layers) - 1, '>>> Last layers should not be changed'
        next_layer = layers[pos+1]

        weak_filter = analyzers[layer].get_weak_filters()
        assert len(weak_filter) < analyzers[layer].num_filter, \
            '>>> Cannot remove the entire output'

        new_arch[layer][0] -= len(weak_filter)
        new_arch[next_layer][1] -= len(weak_filter)
        for flt_idx in weak_filter:
            filter_maps[layer].remove(flt_idx)
            channel_maps[next_layer].remove(flt_idx)

    # sanity check and fix the arch
    for layer in new_arch:
        assert new_arch[layer][0] == len(filter_maps[layer]) and \
            new_arch[layer][1] == len(channel_maps[layer]), \
            '>>> shape mismmatch'
        new_arch[layer] = tuple(new_arch[layer])

    return new_arch, filter_maps, channel_maps

def copy_out(net, new_arch, filter_maps, channel_maps):
    all_weights = {}
    all_biases = {}
    for layer in net.params:
#        print 'Copying out', layer
        old_bias = net.params[layer][1].data
        old_weight = net.params[layer][0].data
        if layer == 'fc6': # fc6 is treated as convolution
            old_shape = old_weight.shape
            old_weight = old_weight.reshape((old_shape[0], -1, 7, 7)) # wired pool5

        # init new weight
        if layer == 'fc8' or layer == 'fc7':
            new_weight_shape = new_arch[layer]
        else:
            new_weight_shape = (new_arch[layer][0], new_arch[layer][1],
                                old_weight.shape[2], old_weight.shape[3])
        new_weight = np.zeros(new_weight_shape, dtype=np.float32)
        new_bias = np.zeros(new_weight_shape[:1], dtype=np.float32)

        fmap = filter_maps[layer]
        cmap = channel_maps[layer]
        # copy out
        new_bias[...] = old_bias[fmap]
        for new_fidx, old_fidx in enumerate(fmap):
            new_weight[new_fidx][...] = old_weight[old_fidx][cmap]

        if layer == 'fc6':
            num_filter = new_weight.shape[0]
            filter_size = new_weight[0].size
            new_weight = new_weight.reshape((num_filter, filter_size))

        all_weights[layer] = new_weight
        all_biases[layer] = new_bias
    return all_weights, all_biases

def init_weight(arch, weights, biases, new_model_folder):
    vgg16 = VGG16Model(arch, new_model_folder)
    print 'compression_rate:', vgg16.compresssion_rate
    net = load_model(ptx=vgg16.train_test_ptx, model=None)

    assert arch.keys() == net.params.keys()

    for layer in net.params:
#        print layer
#        print weights[layer].shape
        net.params[layer][0].data[...] = weights[layer]
        net.params[layer][1].data[...] = biases[layer]

    net_path = os.path.join(vgg16.loc, 'init_weight')
    print 'Saving init weight into:', net_path
    net.save(net_path)
    return net_path

def load_and_eval(arch, model, model_folder, layers_of_interest):
    print 'arch:', arch
    vgg16 = VGG16Model(arch, model_folder)
    net = load_model(vgg16.train_test_ptx, model)

    analyzers_file = os.path.join(vgg16.loc, 'analyzers.pkl')
    if os.path.exists(analyzers_file):
        analyzers = cPickle.load(file(analyzers_file, 'rb'))
    else:
        analyzers = OrderedDict()
        for layer in arch.keys():
            if layer == 'fc8':
                continue
            analyzers[layer] = LayerAnalyzer(layer, arch[layer][0])
        print 'Analyzers:', analyzers.keys()

        acc1, acc5 = 0, 0
        num_forward = 1000
        for i in range(num_forward):
            res = net.forward()
            if i % 50 == 0:
                print i, 'out of', num_forward
            acc1 += res['accuracy@1'].sum() / num_forward
            acc5 += res['accuracy@5'].sum() / num_forward
            for layer in analyzers:
                analyzers[layer].eval_output(net)
        print acc1, acc5
        cPickle.dump(analyzers, file(analyzers_file, 'w'))

#    print layers_of_interest
#    loi_analyzers = dict((layer, analyzers[layer]) for layer in layers_of_interest)
#    new_arch, filter_maps, channel_maps = trim_filters(arch, loi_analyzers)
#    weights, biases = copy_out(net, new_arch, filter_maps, channel_maps)
#    return new_arch, weights, biases

if __name__ == '__main__':

    model_folder = 'vgg16/conv5_3-fc6-trim2-test-trim'
    arch_file = os.path.join(model_folder, 'arch')
    arch = read_arch_from_file(arch_file)
    model = os.path.join(model_folder, 'vgg16_iter_8000.caffemodel')

#--------------------------
#    arch = generate_vgg16_arch()
#    model = os.path.join(model_folder, 'VGG_ILSVRC_16_layers.caffemodel')

    lois = ['conv5_1', 'conv5_2', 'conv5_3', 'fc6']

    load_and_eval(arch, model, model_folder, lois)

#    new_arch, weights, biases = load_and_eval(arch, model, model_folder, lois)
#    weight_init = init_weight(new_arch, weights, biases, 'vgg16/conv5-fc6-trim3')

#    print 'compression_rate: ', calc_size(generate_vgg16_arch())/float(calc_size(new_arch))
