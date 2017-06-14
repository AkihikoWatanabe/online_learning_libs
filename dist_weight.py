# coding=utf-8

import numpy as np
import scipy.sparse as sp
import gzip
import cPickle

class DistWeight():
    """ Class for weight parameter for AROW and SCW.
    """

    def __init__(self, dims=100000):
        """
        Params:
            dims(int): # of dimension for weight vector (default:100000)
        """
        self.dims = dims
        self.mu = sp.csr_matrix((1, dims), dtype=np.float32) # weight parameter
        self.sigma = sp.csr_matrix(([1.0 for _ in xrange(dims)], ([0 for _ in xrange(dims)], range(dims))), (1, dims), dtype=np.float32) # confidence parameter
        #self.sigma = sp.csr_matrix(([1.0 for _ in xrange(dims)], (range(dims), range(dims))), (dims, dims)) # confidence parameter
        self.epoch = 0

    def set_weight(self, new_weight):
        """
        Params:
            new_weight(csr_marix): new weight parameter to set
        """
        self.mu = new_weight

    def get_weight(self):

        return self.mu

    def set_conf(self, new_confidence):
        """
        Params:
            new_confidence(csr_marix): new confidence parameter to set
        """
        self.sigma = new_confidence

    def get_conf(self):

        return self.sigma

    def dump_weight(self, path):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
        """        
        np.savez(path+".epoch{epoch_num}".format(epoch_num=self.epoch), \
                mu_data=self.mu.data, \
                mu_indices=self.mu.indices, \
                mu_indptr=self.mu.indptr, \
                sigma_data=self.sigma.data, \
                sigma_indices=self.sigma.indices, \
                sigma_indptr=self.sigma.indptr, \
                dims=[self.dims])

    def load_weight(self, path, epoch):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
            epoch(int): number of epochs
        """        
        data = np.load(path+".epoch{epoch_num}.npz".format(epoch_num=epoch))
        dims = data["dims"][0]
        self.mu = sp.csr_matrix((data["mu_data"], \
                                 data["mu_indices"], \
                                 data["mu_indptr"]), (1, dims))
        self.sigma = sp.csr_matrix((data["sigma_data"], \
                                 data["sigma_indices"], \
                                 data["sigma_indptr"]), (1, dims))
        self.epoch = epoch

    def extend_weight_dims(self, dims):
        """ Extend # of dimensions on weight vector.
        Params:
            dims(int): # of dimensions to extend
        """
        self.mu = sp.csr_matrix((self.mu.data, self.mu.indices, self.mu.indptr), (1, dims))
        #self.sigma = sp.csr_matrix(np.concatenate(self.sigma.toarray()[0] + \
        #        [1.0 for _ in xrange(dims-self.dims)]))
        self.sigma = sp.csr_matrix(([1.0 for _ in xrange(dims)], (range(dims), range(dims))), (1, dims)) # confidence parameter
        self.dims = dims
