from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix
import numpy as np

class SoftmaxRegression(NNBase):
    """
    Dummy example, to show how to implement a network.
    This implements softmax regression, trained by SGD.
    """

    def __init__(self, dims=[100, 5],
                 reg=0.1, alpha=0.001,
                 rseed=10):
        """
        Set up classifier: parameters, hyperparameters
        """
        ##
        # Store hyperparameters
        self.lreg = reg # regularization
        self.alpha = alpha # default learning rate
        self.nclass = dims[1] # number of output classes
        self.dims = dims # todo move to superclass

        ##
        # NNBase stores parameters in a special format
        # for efficiency reasons, and to allow the code
        # to automatically implement gradient checks
        # and training algorithms, independent of the
        # specific model architecture
        # To initialize, give shapes as if to np.array((m,n))
        param_dims = dict(W = (dims[1], dims[0]), # 5x100 matrix
                          b = (dims[1])) # column vector
        # These parameters have sparse gradients,
        # which is *much* more efficient if only a row
        # at a time gets updated (e.g. word representations)
        NNBase.__init__(self, param_dims)

        ##
        # Now we can access the parameters using
        # self.params.<name> for normal parameters
        # self.sparams.<name> for params with sparse gradients
        # and get access to normal NumPy arrays
        self.params.W = random_weight_matrix(*self.params.W.shape)
        # self.params.b1 = zeros((self.nclass,1)) # done automatically!

    def _acc_grads(self, x, label):
        """
        Accumulate gradients from a training example.
        """
        ##
        # Forward propagation
        #import ipdb; ipdb.set_trace()
        p = softmax(self.params.W.dot(x) + self.params.b)

        ##
        # Compute gradients w.r.t cross-entropy loss
        y = make_onehot(label, len(p))
        delta = p - y
        # dJ/dW, dJ/db1
        self.grads.W += np.outer(delta, x) + self.lreg * self.params.W
        self.grads.b += delta
        # dJ/dL, sparse update: use sgrads
        # this stores an update to the row L[idx]
        #self.sgrads.L[idx] = self.params.W.T.dot(delta)
        # note that the syntax is overloaded here; L[idx] =
        # works like +=, so if you update the same index
        # twice, it'll store *BOTH* updates. For example:
        # self.sgrads.L[idx] = ones(50)
        # self.sgrads.L[idx] = ones(50)
        # will add -2*alpha to that row when gradients are applied!

        ##
        # We don't need to do the update ourself, as NNBase
        # calls that during training. See NNBase.train_sgd
        # in nn/base.py to see how this is done, if interested.
        ##

    def compute_loss_full(self, X, y):
        #import ipdb; ipdb.set_trace()
        loss = 0
        for idx in xrange(X.shape[0]):
            loss += self.compute_loss(X[idx], y[idx])
        return loss


    def compute_loss(self, x, label):
        """
        Compute the cost function for a single example.
        """
        #import ipdb; ipdb.set_trace()
        ##
        # Forward propagation
        p = softmax(self.params.W.dot(x) + self.params.b)
        J = -1*np.log(p[label]) # cross-entropy loss
        Jreg = (self.lreg / 2.0) * np.sum(self.params.W**2.0)
        return J + Jreg
    
    def predict_proba(self, X):
        # we can guess numb of class by params.b
        return np.apply_along_axis(self.predict_proba_single, 1, X)

    def predict_proba_single(self, x):
        """
        Predict class probabilities.
        """
        p = softmax(self.params.W.dot(x) + self.params.b)
        return p
    
    def predict(self, X):
        return np.apply_along_axis(self.predict_single, 1, X)

    def predict_single(self, x):
        """Predict most likely class."""
        P = self.predict_proba_single(x)
        return np.argmax(P)
    
    def get_weights(self):
        W = np.hstack([self.params.W, self.params.b.reshape(-1, 1)])
        return (W, )

