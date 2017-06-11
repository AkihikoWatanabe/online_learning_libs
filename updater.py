# coding=utf-8

"""
This is the python implementation of the online learning method using Iterative Parameter Mixture.
This implementation is now supporting:
    - Perceptron
    - PA-I, PA-II
    - AROW
"""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from update_func import Perceptron, PA_I, PA_II, AROW

class Updater():
    """ This class support some online learning methods, i.e. weight update method, using Iterative Parameter Mixture.
    """

    def __init__(self, C=0.01, r=1.0, process_num=1, method="PA-II"):
        """ 
        Params:
            C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
            r(float): regularization parameter for AROW
            process_num(int): # of parallerization (default:1)
            method(str): learning method (Perceptrion, PA-I, PA-II, AROW)
            """ 
        self.C = C # Parameter to adjust the degree of penalty on PA-II (C>=0)
        self.r = r # regularization parameter for AROW  
        self.PROCESS_NUM = process_num 
        self.METHOD = method # default PA-II
        assert self.METHOD in ["Perceptron", "PA-I", "PA-II", "AROW"], "Invalid method name {name}".format(self.METHOD)

    def __make_minibatch(self, x_list, y_list):
        """
        Params:
            x_list(csr_matrix): csr_matrix of feature vectors.
            y_list(list): np.ndarray of labels corresponding to each feature vector
        Returns:
            x_batch(list): batch of feature vectors
            y_batch(list): batch of labels
        """

        x_batch = []
        y_batch = []
        N = x_list.shape[0] # # of data
        np.random.seed(0) # set seed for permutation
        perm = np.random.permutation(N)

        for p in xrange(self.PROCESS_NUM):
            ini = N * (p) / self.PROCESS_NUM
            fin = N * (p + 1) / self.PROCESS_NUM
            x_batch.append(x_list[perm[ini:fin]])
            y_batch.append(y_list[perm[ini:fin]])

        return x_batch, y_batch

    def __iterative_parameter_mixture(self, callback, weight):
        """
        Params:
            callback: callback for parallerized process
            weight(Weight): current weight class
        Returns:
            loss_list(list): list of loss value
        """
        _w_sum = sp.csr_matrix((1, weight.dims), dtype=np.float32)
        loss_list = []
        for _w, _loss_list in callback:
            _w_sum += _w
            loss_list += _loss_list

        # insert updated weight
        weight.set_weight(1.0 / self.PROCESS_NUM * _w_sum)
        weight.epoch += 1

        return loss_list

    def update(self, x_list, y_list, weight):
        """ Update weight parameter according to PA-II update rule.
        Params:
            x_list(csr_matrix): csr_matrix of feature vectors.
            y_list(list): np.ndarray of labels corresponding to each feature vector
            weight(Weight, DistWeight): class of weight (if you want to use AROW, weight class should be DistWeight)
        Returns:
            loss_list(list): List of loss value
        """
        assert x_list.shape[0] == len(y_list), "invalid shape: x_list, y_list"
        
        # make minibatch for Iterative Parameter Mixture
        if self.METHOD != "AROW":
            x_batch, y_batch = self.__make_minibatch(x_list, y_list)
        
        # choose learning method and run
        if self.METHOD == "Perceptron":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(Perceptron)(i, x_batch[i], y_batch[i], weight.get_weight()) for i in range(self.PROCESS_NUM)) 
            loss_list = self.__iterative_parameter_mixture(callback, weight)
        elif self.METHOD == "PA-I":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(PA_I)(i, x_batch[i], y_batch[i], weight.get_weight(), self.C) for i in range(self.PROCESS_NUM)) 
            loss_list = self.__iterative_parameter_mixture(callback, weight)
        elif self.METHOD == "PA-II":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(PA_II)(i, x_batch[i], y_batch[i], weight.get_weight(), self.C) for i in range(self.PROCESS_NUM)) 
            loss_list = self.__iterative_parameter_mixture(callback, weight)
        elif self.METHOD == "AROW":
            loss_list, mu, sigma = AROW(x_list, y_list, weight.get_weight(), weight.get_conf(), self.r)
            weight.set_weight(mu)
            weight.set_conf(sigma)
            weight.epoch += 1

        return loss_list
