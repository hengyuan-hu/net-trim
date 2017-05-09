import sys, os

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from ascii_hist import Histogram

def _criteria_func(zero_rates):
    return np.where(zero_rates > np.mean(zero_rates) + np.std(zero_rates))[0]

class LayerAnalyzer:
    def __init__(self, name, num_filter, criteria_func=_criteria_func):
        self.name = name
        self.num_filter = num_filter
        self.zero_rates = np.zeros((self.num_filter,))
        self.fwd_count = 0
        self.criteria_func = criteria_func

    def hist_plot(self):
        his = Histogram(self.zero_rates, bins=50)
        print his.vertical()

    def thres_plot(self):
        data = self.zero_rates
        start = 1.0 * int(data.min() * 20) / 20
        end = 1.0 * int((data.max() + 0.01) * 20) / 20
        thres = np.arange(start, end, 0.05)
        print 'total of', data.size, 'filters'
        for th in thres:
            num = (np.where(data >= th)[0]).size
            print '+%4.02f' % th,
            print num, '\t', float(num) / data.size * 100, '%'
      
    def eval_output(self, net):
        blob_data = net.blobs[self.name].data
        assert self.num_filter == blob_data.shape[1], \
            '>>> Shape mismatch: {0} vs {1}'.format(self.num_filter, 
                                                    blob_data.shape[1])
        for i in range(self.num_filter):
            if 'fc' in self.name:
                output = blob_data[:, i]
            else:
                output = blob_data[:, i, :, :]

            zr = (output <= 1e-9).sum() / float(output.size)
            self.zero_rates[i] = (zr + self.zero_rates[i] * self.fwd_count) / float(self.fwd_count+1)
        self.fwd_count += 1

    def get_weak_filters(self):
        self.to_remove = self.criteria_func(self.zero_rates)
        return self.to_remove
