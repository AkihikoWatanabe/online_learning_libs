# coding=utf-8

"""
This is the python implementation of the online learning method using Iterative Parameter Mixture.
This implementation is now supporting:
    - Perceptron
    - PA-I, PA-II
"""

import numpy as np
import cPickle
import gzip
from joblib import Parallel, delayed

from update_func import Perceptron, PA_I, PA_II

class Updater():
    """ This class support some online learning methods, i.e. weight update method, using Iterative Parameter Mixture.
    """

    def __init__(self, C=0.01, process_num=1, method="PA-II"):
        """ 
        Params:
            C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
            process_num(int): # of parallerization (default:1)
            method(str): learning method (Perceptrion, PA-I, PA-II)
            """ 
        self.C = C # Parameter to adjust the degree of penalty on PA-II (C>=0)
        self.PROCESS_NUM = process_num 
        self.METHOD = method # default PA-II
        assert self.METHOD in ["Perceptron", "PA-I", "PA-II"], "Invalid method name {name}".format(self.METHOD)

    def __make_minibatch(self, x_list, y_list):
        x_batch = []
        y_batch = []
        N = len(x_list) # # of data
        perm = np.random.permutation(N)

        for p in xrange(self.PROCESS_NUM):
            ini = N * (p) / self.PROCESS_NUM
            fin = N * (p + 1) / self.PROCESS_NUM
            x_batch.append(x_list[perm[ini:fin]])
            y_batch.append(y_list[perm[ini:fin]])

        return x_batch, y_batch

    def update(self, x_list, y_list, weight):
        """ Update weight parameter according to PA-II update rule.
        Params:
            x_list(list): List of feature vectors. Each vector is represented by np.ndarray.
            y_list(list): List of labels corresponding to each feature vector.
        Returns:
            loss_list(list): List of loss value
        """
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        assert len(x_list) == len(y_list), "invalid length: x_list, y_list"
        
        # make minibatch for Iterative Parameter Mixture
        x_batch, y_batch = self.__make_minibatch(x_list, y_list)
        
        # choose learning method and run
        if self.METHOD == "Perceptron":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(Perceptron)(i, x_batch[i], y_batch[i], np.array(weight.w)) for i in range(self.PROCESS_NUM)) 
        elif self.METHOD == "PA-I":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(PA_I)(i, x_batch[i], y_batch[i], np.array(weight.w), self.C) for i in range(self.PROCESS_NUM)) 

        elif self.METHOD == "PA-II":
            callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                    delayed(PA_II)(i, x_batch[i], y_batch[i], np.array(weight.w), self.C) for i in range(self.PROCESS_NUM)) 

        # Iterative Parameter Mixture
        _w_sum = np.asarray([0.0 for _ in xrange(len(weight.w))], dtype=np.float32)
        loss_list = []
        for _w, _loss_list in callback:
            _w_sum += _w
            loss_list += _loss_list

        # insert updated weight
        weight.w =  1.0 / self.PROCESS_NUM * _w_sum
        weight.epoch += 1

        return loss_list
