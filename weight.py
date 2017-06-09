# coding=utf-8

import numpy as np
import scipy.sparse as sp
import gzip
import cPickle

class Weight():

    def __init__(self, dims=100000):
        """
        Params:
            dims(int): # of dimension for weight vector (default:100000)
        """
        self.dims = dims
        self.w = sp.csr_matrix((1, dims), dtype=np.float32) # weight parameter
        self.epoch = 0

    def dump_weight(self, path):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
        """        
        np.save(path+".epoch{epoch_num}".format(epoch_num=self.epoch), self.w)

    def load_weight(self, path, epoch):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
            epoch(int): number of epochs
        """        
        self.w = np.load(path+".epoch{epoch_num}".format(epoch_num=epoch))
        self.epoch = epoch

    def extend_weight_dims(self, dims):
        """ Extend # of dimensions on weight vector.
        Params:
            dims(int): # of dimensions to extend
        """
        self.w = sp.csr_matrix((self.w.data, self.w.indices, self.w.indptr), (1, dims))
        self.dims = dims
