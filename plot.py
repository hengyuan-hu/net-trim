import os, sys
from eval_net import LayerAnalyzer
import cPickle

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'usage'
        print 'python blah.py conv5_3-fc6-... conv5_3+fc6 thres/hist'
        sys.exit()

    model = sys.argv[1]
    layers = sys.argv[2].split('+')
    mode = sys.argv[3]
    assert mode in ['thres', 'hist']
    
    with open(os.path.join(model, 'analyzers.pkl'), 'r') as f:
        analyzers = cPickle.load(f)

    for layer in layers:
        print layer
        print 
        if mode == 'hist':
            analyzers[layer].hist_plot()
        elif mode == 'thres':
            analyzers[layer].thres_plot()
        print '========================='
    
