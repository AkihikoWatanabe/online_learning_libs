# coding=utf-8

def hinge_loss(x, y, w):
    """ Calculate hinge loss.
    Params:
        x(np.ndarray): feature vector
        y(int): label
        w(np.ndarray): weight vector
    Returns:
        hinge(float): hinge loss
    """
   
    return max(0.0, 1.0 - y * w.dot(x))

def Perceptron(i, x_batch, y_batch, w):
    """ Update weight parameter using Perceptrion update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(np.ndarray): minibatch of feature vector
        y_batch(np.ndarray): minibatch of labels
        w(np.ndarray): current weight parameters
    """
    
    for x, y in zip(x_batch, y_batch):
        if y * w.dot(x) < 0:
            w += y * x
    return w, None 

def PA_I(i, x_batch, y_batch, w, C):
    """ Update weight parameter using PA-I update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(np.ndarray): minibatch of feature vector
        y_batch(np.ndarray): minibatch of labels
        w(np.ndarray): current weight parameters
        C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
    """
    loss_list = []
    for x, y in zip(x_batch, y_batch):
        loss = hinge_loss(x, y, w)
        loss_list.append(loss)
        if loss > 0.0:
            w += min(C, hinge_loss(x, y, w) / x.dot(x)) * y * x
    return w, loss_list

def PA_II(i, x_batch, y_batch, w, C):
    """ Update weight parameter using PA-II update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(np.ndarray): minibatch of feature vector
        y_batch(np.ndarray): minibatch of labels
        w(np.ndarray): current weight parameters
        C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
    """
    loss_list = []
    for x, y in zip(x_batch, y_batch):
        loss = hinge_loss(x, y, w)
        loss_list.append(loss)
        if loss > 0.0:
            w += (hinge_loss(x, y, w) / (x.dot(x) + 1.0 / (2.0 * C))) * y * x
    return w, loss_list
