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
from Joblib import Parallel, delayed

from update_func import Perceptron, PA_I, PA_II

class Updater():
    """ This class support some online learning methods, i.e. weight update method, using Iterative Parameter Mixture.
    """

    def __init__(self, C, process_num=1, method="PA-II"):
        """ 
        Params:
            weight(Weight): # of dimension for weight vector
            C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
            process_num(int): # of parallerization (default:1)
            method(str): learning method (Perceptrion, PA-I, PA-II)
            """ 
        self.C = C # Parameter to adjust the degree of penalty on PA-II (C>=0)
        self.PROCESS_NUM = process_num 
        self.method = method # default PA-II
        assert self.method in ["Perceptron", "PA-I", "PA-II"], "Invalid method name {name}".format(self.method)

    def __make_minibatch(x_list, y_list):
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
        """
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)
        assert len(x_list) == len(y_list), "invalid length: x_list, y_list"
        
        # make minibatch for Iterative Parameter Mixture
        x_batch, y_batch = self.__make_minibatch(x_list, y_list)
        
        # choose learning method and run
        if self.type == "Perceptron":
            callback = Parallel(n_jobs=self.process_num)( \
                    delayed(Perceptron)(i, x_batch[i], y_batch[i], np.array(weight.w)) for i in range(self.PROCESS_NUM)) 
        elif self.type == "PA-I":
            callback = Parallel(n_jobs=self.process_num)( \
                    delayed(PA_I)(i, x_batch[i], y_batch[i], np.array(weight.w), self.C) for i in range(self.PROCESS_NUM)) 

        elif self.type == "PA-II":
            callback = Parallel(n_jobs=self.process_num)( \
                    delayed(PA_II)(i, x_batch[i], y_batch[i], np.array(weight.w), self.C) for i in range(self.PROCESS_NUM)) 

        # Iterative Parameter Mixture
        _w_sum = np.asarray([0.0 for _ in xrange(len(weight.w))], dtype=np.float32)
        for _w in callback:
            _w_sum += _w

        # insert updated weight
        weight.w =  1.0 / self.PROCESS_NUM * _w_sum
        weight.epoch += 1
