# coding=utf-8

import numpy as np
import scipy.sparse as sp

def hinge_loss(x, y, w):
    """ Calculate hinge loss.
    Params:
        x(csr_matrix): feature vector
        y(int): label
        w(csr_matrix): weight vector
    Returns:
        hinge(float): hinge loss
    """
    # hinge = max(0.0, 1.0 - y * w^{T}x)
    return max(0.0, 1.0 - y * w.multiply(x).sum()) 

def Perceptron(i, x_batch, y_batch, w):
    """ Update weight parameter using Perceptrion update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(csr_matrix): minibatch of feature vector
        y_batch(list): minibatch of labels
        w(csr_matrix): current weight parameters
    Returns:
        w(csr_matrix): updated parameter
    """
   
    for j in xrange(x_batch.shape[0]):
        # y * w^{T} * x
        if y_batch[j] * w.multiply(x_batch[j]).sum() < 0:
            # w^{t+1} = w^{t} + y * x
            w += y_batch[j] * x_batch[j]
    return w, None 

def PA_I(i, x_batch, y_batch, w, C):
    """ Update weight parameter using PA-I update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(csr_matrix): minibatch of feature vector
        y_batch(list): minibatch of labels
        w(csr_matrix): current weight parameters
        C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
    Returns:
        w(csr_matrix): updated parameter
        loss_list(list): list of loss value
    """
    loss_list = []
    for j in xrange(x_batch.shape[0]):
        loss = hinge_loss(x_batch[j], y_batch[j], w)
        loss_list.append(loss)
        if loss > 0.0:
            # w^{t+1} = w^{t} + min(C, hinge_loss(x, y, w) / norm(x)) * y * x
            w += min(C, loss / x_batch[j].multiply(x_batch[j]).sum()) * y_batch[j] * x_batch[j]
    return w, loss_list

def PA_II(i, x_batch, y_batch, w, C):
    """ Update weight parameter using PA-II update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(csr_matrix): minibatch of feature vector
        y_batch(list): minibatch of labels
        w(csr_matrix): current weight parameters
        C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
    Returns:
        w(csr_matrix): updated parameter
        loss_list(list): list of loss value
    """
    loss_list = []
    for j in xrange(x_batch.shape[0]):
        loss = hinge_loss(x_batch[j], y_batch[j], w)
        loss_list.append(loss)
        if loss > 0.0:
            # w^{t+1} = w^{t} + hinge_loss(x, y, w) / (norm(x)+ 1/2C) * y * x
            w += (loss / (x_batch[j].multiply(x_batch[j]).sum() + 1.0 / (2.0 * C))) * y_batch[j] * x_batch[j]
    return w, loss_list

def AROW(x_list, y_list, mu, sigma, r):
    """ Update weight parameter using AROW update rule on the given data.
    Params:
        x_list(csr_matrix): list of feature vector
        y_list(list): list of labels
        mu(csr_matrix): current weight parameters
        sigma(csr_matrix): current confidence parameters
        r(float): regularization parameter
    Returns:
        mu(csr_matrix): updated parameter
        sigma(csr_matrix): updated confidence
        loss_list(list): list of loss value
    """
    loss_list = []
    for j in xrange(x_list.shape[0]):
        # calculate margin
        m = mu.multiply(x_list[j]).sum()
        # calculate confidence
        # sigma * x 
        cx = sp.csr_matrix(sigma.multiply(x_list[j]).T.sum(axis=0))
        v = cx.multiply(x_list[j]).sum()
        #loss = 1.0 if np.sign(m)!=np.sign(y_list[j]) else 0.0
        loss = hinge_loss(x_list[j], y_list[j], mu)
        loss_list.append(loss)
        if m * y_list[j] < 1.0:
            beta = 1.0 / (v + r)
            alpha = loss * beta
            mu += alpha * y_list[j] * cx
            # sigma * x
            _cx = sp.csr_matrix(sigma.multiply(x_list[j]).sum(axis=1))
            # x^{T} * sigma
            _xc = x_list[j].dot(sigma)
            sigma += -beta * _cx.dot(_xc)
    return loss_list, mu, sigma
