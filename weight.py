# coding=utf-8

import numpy as np
import gzip
import cPickle

class Weight():

    def __init__(self, dims=100000):
        """
        Params:
            weight(Weight): # of dimension for weight vector (default:100000)
        """
        self.dims = dims
        self.w = np.asarray([0.0 for _ in xrange(self.dims)], dtype=np.float32) # weight parameter
        self.epoch = 0

    def dump_weight(self, path):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
        """        
        with gzip.open(path+".epoch{epoch_num}.pkl.gz".format(epoch_num=self.epoch), 'wb') as gf:
            cPickle.dump(self.w, gf, cPickle.HIGHEST_PROTOCOL)

    def load_weight(self, path, epoch):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
            epoch(int): number of epochs
        """        
        with gzip.open(file_path+".epoch{epoch_num}".format(epoch_num=self.epoch), 'rb') as gf:
            self.w = cPickle.load(gf)
        self.epoch = epoch

    def extend_weight_dims(self, dims):
        """ Extend # of dimensions on weight vector.
        Params:
            dims(int): # of dimensions to extend
        """
        self.w = np.concatenate((self.w, np.asarray([0.0 for _ in xrange(dims)], dtype=np.float32)), axis=0)

