# -*- coding: utf-8 -*-
import numpy as np

class Histogram(object):
    """
    Ascii histogram
    """
    def __init__(self, data, bins=10):
        """
        Class constructor
        
        :Parameters:
        - `data`: array like object
        """
        self.data = data
        self.bins = bins
        self.h = np.histogram(self.data, bins=self.bins)

    def vertical(self):
        """
        Returns a Multi-line string containing a
        a vertical histogram representation of self.data
        :Parameters:
        - `height`: Height of the histogram in characters
        - `character`: Character to use
        >>> d = normal(size=1000)
        >>> Histogram(d,bins=10)
        >>> print h.vertical(15,'*')
        236
        -3.42:
        -2.78:
        -2.14: ***
        -1.51: *********
        -0.87: *************
        -0.23: ***************
        0.41 : ***********
        1.04 : ********
        1.68 : *
        2.32 :
        """
        his = ""
        MAX_LEN = 50.0
        max_freq = self.h[0].max()
        his += '%+4.2f ' % self.h[1][0]
        for i in range(self.bins):
            percentage = 1.0 * self.h[0][i] / self.h[0].sum()
            accum_percentage = 1.0 * self.h[0][:i].sum() / self.h[0].sum()
            his += '\n'
            his += '%+4.2f ' % self.h[1][i+1]
            his += '%5.1f ' % (100 - accum_percentage * 100)
            his += '*' * int(MAX_LEN * self.h[0][i] / max_freq)
            his += '  %.1f' % (percentage * 100)
        his += '\n'
        return his

if __name__ == "__main__":
    from numpy.random import normal
    d = normal(size=1000)
    h = Histogram(d,bins=20)
    print h.vertical()
