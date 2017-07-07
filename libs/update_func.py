# coding=utf-8

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

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

def scw_loss(phi, v, m):
    """ Calculate loss for SCW.
    Params:
        phi(float): parameter that derived from cumulative normal dist PHI(eta)
        v(float): confidence
        m(float): margin
    """
    return max(0, phi * np.sqrt(v) - m)

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

def CW(x_list, y_list, mu, sigma, eta):
    """ Update weight parameter using SCW-I update rule on the given data.
    Params:
        x_list(csr_matrix): list of feature vector
        y_list(list): list of labels
        mu(csr_matrix): current weight parameters
        sigma(csr_matrix): current confidence parameters
        eta(float): confidence parameter 
    Returns:
        mu(csr_matrix): updated parameter
        sigma(csr_matrix): updated confidence
    """
    phi = norm.cdf(eta) # cumulative function over normal dist
    psi = 1.0 + phi ** 2 / 2.0
    zeta = 1.0 + phi ** 2
    for j in xrange(x_list.shape[0]):
        # margin
        m = y_list[j] * mu.multiply(x_list[j]).sum()
        # confidence
        cx = sigma.multiply(x_list[j])
        v = cx.multiply(x_list[j]).sum()

        alpha = max(0.0, 1.0 / (v * zeta) * \
                (-m * psi + np.sqrt(m ** 2 * phi ** 4 / 4.0 + v * phi ** 2 * zeta)))
        u = 1.0 / 4.0 * (-alpha * v * phi + \
                np.sqrt(alpha ** 2 * v ** 2 * phi ** 2 + 4.0 * v)) ** 2
        beta = alpha * phi / (np.sqrt(u) + v * alpha * phi)
        # update mu
        mu += alpha * y_list[j] * cx
        # update sigma
        sigma += -beta * cx.multiply(cx)
    return None, mu, sigma

def AROW(x_list, y_list, mu, sigma, r):
    """ Update weight parameter using AROW update rule on the given data.
    Params:
        x_list(csr_matrix): list of feature vector
        y_list(list): list of labels
        mu(csr_matrix): current weight parameters
        sigma(csr_matrix): current confidence parameters
        r(float): regularization parameter
    Returns:
        loss_list(list): list of loss value
        mu(csr_matrix): updated parameter
        sigma(csr_matrix): updated confidence
    """
    loss_list = []
    for j in xrange(x_list.shape[0]):
        # calculate margin
        m = mu.multiply(x_list[j]).sum()
        # calculate confidence
        # sigma * x 
        cx = sigma.multiply(x_list[j])
        v = cx.multiply(x_list[j]).sum()
        #loss = 1.0 if np.sign(m)!=np.sign(y_list[j]) else 0.0
        loss = hinge_loss(x_list[j], y_list[j], mu)
        loss_list.append(loss)
        if m * y_list[j] < 1.0:
            beta = 1.0 / (v + r)
            alpha = loss * beta
            # update mu
            mu += alpha * y_list[j] * cx
            # update sigma
            sigma += -beta * cx.multiply(cx)
    return loss_list, mu, sigma

def SCW_I(x_list, y_list, mu, sigma, C, eta):
    """ Update weight parameter using SCW-I update rule on the given data.
    Params:
        x_list(csr_matrix): list of feature vector
        y_list(list): list of labels
        mu(csr_matrix): current weight parameters
        sigma(csr_matrix): current confidence parameters
        C: aggressive parameter
        eta(float): parameter for cumulative function of normal dist
    Returns:
        loss_list(list): list of loss value
        mu(csr_matrix): updated parameter
        sigma(csr_matrix): updated confidence
    """
    loss_list = []
    phi = norm.cdf(eta) # cumulative function over normal dist
    psi = 1.0 + phi ** 2 / 2.0
    zeta = 1.0 + phi ** 2
    for j in xrange(x_list.shape[0]):
        # margin
        m = y_list[j] * mu.multiply(x_list[j]).sum()
        # confidence
        cx = sigma.multiply(x_list[j])
        v = cx.multiply(x_list[j]).sum()
        loss = scw_loss(phi, v, m) 
        loss_list.append(loss)
        if loss > 0.0:
            alpha = min(C, max(0.0, 1.0 / (v * zeta) * \
                    (-m * psi + np.sqrt(m ** 2 * phi ** 4 / 4.0 + v * phi ** 2 * zeta))))
            u = 1.0 / 4.0 * (-alpha * v * phi + \
                    np.sqrt(alpha ** 2 * v ** 2 * phi ** 2 + 4.0 * v)) ** 2
            beta = alpha * phi / (np.sqrt(u) + v * alpha * phi)
            # update mu
            mu += alpha * y_list[j] * cx
            # update sigma
            sigma += -beta * cx.multiply(cx)
    return loss_list, mu, sigma

def SCW_II(x_list, y_list, mu, sigma, C, eta):
    """ Update weight parameter using SCW-I update rule on the given data.
    Params:
        x_list(csr_matrix): list of feature vector
        y_list(list): list of labels
        mu(csr_matrix): current weight parameters
        sigma(csr_matrix): current confidence parameters
        C: aggressive parameter
        eta(float): parameter for cumulative function of normal dist
    Returns:
        loss_list(list): list of loss value
        mu(csr_matrix): updated parameter
        sigma(csr_matrix): updated confidence
    """
    loss_list = []
    phi = norm.cdf(eta) # cumulative function over normal dist
    psi = 1.0 + phi ** 2 / 2.0
    zeta = 1.0 + phi ** 2
    for j in xrange(x_list.shape[0]):
        # margin
        m = y_list[j] * mu.multiply(x_list[j]).sum()
        # confidence
        cx = sigma.multiply(x_list[j])
        v = cx.multiply(x_list[j]).sum()
        loss = scw_loss(phi, v, m) 
        loss_list.append(loss)
        if loss > 0.0:
            n = v + 1.0 / (2.0 * C)
            ganma = phi * np.sqrt(phi ** 2 * m ** 2 * v ** 2 + \
                    4.0 * n * v * (n + v * phi ** 2))
            alpha = max(0.0, (-(2.0 * m * n + phi ** 2 * m * v) + ganma) / \
                    2.0 * (n ** 2 + n * v * phi ** 2))
            u = 1.0 / 4.0 * (-alpha * v * phi + \
                    np.sqrt(alpha ** 2 * v ** 2 * phi ** 2 + 4.0 * v)) ** 2
            beta = alpha * phi / (np.sqrt(u) + v * alpha * phi)
            # update mu
            mu += alpha * y_list[j] * cx
            # update sigma
            sigma += -beta * cx.multiply(cx)
    return loss_list, mu, sigma
