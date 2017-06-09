# coding=utf-8

def hinge_loss(x, y, w):
    """ Calculate hinge loss.
    Params:
        x(csr_matrix): feature vector
        y(int): label
        w(np.ndarray): weight vector
    Returns:
        hinge(float): hinge loss
    """
    # hinge = max(0.0, 1.0 - y * w^{T}x)
    return max(0.0, 1.0 - y.multiply(w.multiply(x)).sum()) 

def Perceptron(i, x_batch, y_batch, w):
    """ Update weight parameter using Perceptrion update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(np.ndarray): minibatch of feature vector
        y_batch(np.ndarray): minibatch of labels
        w(np.ndarray): current weight parameters
    """
   
    for j in xrange(x_batch.shape[0]):
        # y * w^{T} * x
        if y_batch[j].multiply(w.multiply(x_batch[j])).sum() < 0:
            # w^{t+1} = w^{t} + y * x
            w += y_batch[j].multiply(x_batch[j])
    return w, None 

def PA_I(i, x_batch, y_batch, w, C):
    """ Update weight parameter using PA-I update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(csr_matrix): minibatch of feature vector
        y_batch(csr_matrix): minibatch of labels
        w(csr_matrix): current weight parameters
        C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
    """
    loss_list = []
    for j in xrange(x_batch.shape[0]):
        loss = hinge_loss(x_batch[j], y_batch[j], w)
        loss_list.append(loss)
        if loss > 0.0:
            # w^{t+1} = w^{t} + min(C, hinge_loss(x, y, w) / norm(x)) * y * x
            w += min(C, loss / x_batch[j].multiply(x_batch[j]).sum()) * y_batch[j].multiply(x_batch[j])
    return w, loss_list

def PA_II(i, x_batch, y_batch, w, C):
    """ Update weight parameter using PA-II update rule on the given minibatch.
    Params:
        i(int): process number
        x_batch(csr_matrix): minibatch of feature vector
        y_batch(csr_matrix): minibatch of labels
        w(csr_matrix): current weight parameters
        C(float): Parameter to adjust the degree of penalty, aggressiveness parameter (C>=0)
    """
    loss_list = []
    for j in xrange(x_batch.shape[0]):
        loss = hinge_loss(x_batch[j], y_batch[j], w)
        loss_list.append(loss)
        if loss > 0.0:
            # w^{t+1} = w^{t} + hinge_loss(x, y, w) / (norm(x)+ 1/2C) * y * x
            w += (loss / (x_batch[j].multiply(x_batch[j]).sum() + 1.0 / (2.0 * C))) * y_batch[j].multiply(x_batch[j])
    return w, loss_list
