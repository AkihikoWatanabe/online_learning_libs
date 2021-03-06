# coding=utf-8

"""
This is the python implementation of the online learning method using Iterative Parameter Mixture.
This implementation is now supporting:
    - Perceptron
    - PA-I, PA-II
    - CW
    - AROW
    - SCW-I, SCW-II
"""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from update_func import Perceptron, PA_I, PA_II, CW, AROW, SCW_I, SCW_II

class Updater():
    """ This class support some online learning methods, i.e. weight update method, using Iterative Parameter Mixture.
    """

    def __init__(self, C=0.01, r=1.0, eta=0.1, process_num=1, method="PA-II"):
        """ 
        Params:
            C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
            r(float): regularization parameter for AROW
            process_num(int): # of parallerization (default:1)
            method(str): learning method (Perceptrion, PA-I, PA-II, CW, AROW, SCW-I, SCW-II)
            """ 
        self.C = C # Parameter to adjust the degree of penalty on PA-II and SCW(C>=0)
        self.r = r # regularization parameter for AROW  
        self.eta = eta # confidence parameter on CW and SCW
        self.PROCESS_NUM = process_num 
        self.METHOD = method # default PA-II
        assert self.METHOD in ["Perceptron", "PA-I", "PA-II", "CW", "AROW", "SCW-I", "SCW-II"], "Invalid method name {name}".format(self.METHOD)

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

    def __iterative_parameter_mixture_for_distweight(self, callback, weight):
        """
        Params:
            callback: callback for parallerized process
            weight(Weight): current weight class
        Returns:
            loss_list(list): list of loss value
        """
        _mu_sum = sp.csr_matrix((1, weight.dims), dtype=np.float32)
        _sigma_sum = sp.csr_matrix(([1.0 for _ in xrange(weight.dims)], ([0 for _ in xrange(weight.dims)], range(weight.dims))), (1, weight.dims), dtype=np.float32)

        loss_list = []
        for _loss_list, _mu, _sigma in callback:
            _mu_sum += _mu
            _sigma_sum += _sigma
            loss_list += _loss_list

        # insert updated weight
        weight.set_weight(1.0 / self.PROCESS_NUM * _mu_sum)
        weight.set_conf(1.0 / self.PROCESS_NUM * _sigma_sum)
        weight.epoch += 1

        return loss_list

    def update(self, x_list, y_list, weight):
        """ Update weight parameter according to {self.METHOD} update rule.
        Params:
            x_list(csr_matrix): csr_matrix of feature vectors.
            y_list(list): np.ndarray of labels corresponding to each feature vector
            weight(Weight, DistWeight): class of weight (if you want to use AROW, weight class should be DistWeight)
        Returns:
            loss_list(list): List of loss value
        """
        assert x_list.shape[0] == len(y_list), "invalid shape: x_list, y_list"
        
        # make minibatch for Iterative Parameter Mixture
        #if self.METHOD == "Perceptrion" or \
        #   self.METHOD == "PA-I" or \
        #   self.METHOD == "PA-II":
        #    x_batch, y_batch = self.__make_minibatch(x_list, y_list)

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
        elif self.METHOD == "CW":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(CW)(x_batch[i], y_batch[i], weight.get_weight(), weight.get_conf(), self.eta) for i in range(self.PROCESS_NUM)) 
            loss_list = self.__iterative_parameter_mixture_for_distweight(callback, weight)
            """
            loss_list, mu, sigma = CW(x_list, y_list, weight.get_weight(), weight.get_conf(), self.eta)
            weight.set_weight(mu)
            weight.set_conf(sigma)
            weight.epoch += 1
            """
        elif self.METHOD == "AROW":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(AROW)(x_batch[i], y_batch[i], weight.get_weight(), weight.get_conf(), self.r) for i in range(self.PROCESS_NUM)) 
            loss_list = self.__iterative_parameter_mixture_for_distweight(callback, weight)
            """
            loss_list, mu, sigma = AROW(x_list, y_list, weight.get_weight(), weight.get_conf(), self.r)
            weight.set_weight(mu)
            weight.set_conf(sigma)
            weight.epoch += 1
            """
        elif self.METHOD == "SCW-I":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(SCW_I)(x_batch[i], y_batch[i], weight.get_weight(), weight.get_conf(), self.C, self.eta) for i in range(self.PROCESS_NUM)) 
            loss_list = self.__iterative_parameter_mixture_for_distweight(callback, weight)
            """
            loss_list, mu, sigma = SCW_I(x_list, y_list, weight.get_weight(), weight.get_conf(), self.C, self.eta)
            weight.set_weight(mu)
            weight.set_conf(sigma)
            weight.epoch += 1
            """
        elif self.METHOD == "SCW-II":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(SCW_II)(x_batch[i], y_batch[i], weight.get_weight(), weight.get_conf(), self.C, self.eta) for i in range(self.PROCESS_NUM)) 
            loss_list = self.__iterative_parameter_mixture_for_distweight(callback, weight)
            """
            loss_list, mu, sigma = SCW_II(x_list, y_list, weight.get_weight(), weight.get_conf(), self.C, self.eta)
            weight.set_weight(mu)
            weight.set_conf(sigma)
            weight.epoch += 1
            """

        return loss_list
