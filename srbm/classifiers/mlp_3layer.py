from nn.base import NNBase
from nn.math import softmax, make_onehot, tanh, tanhd
from misc import random_weight_matrix
import numpy as np

class MLP(NNBase):
    """
    Dummy example, to show how to implement a network.
    This implements softmax regression, trained by SGD.
    """

    def __init__(self, dims=[100, 20, 20, 5],
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


        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          G = (dims[3], dims[2]),
                          b3 = (dims[3], ),
                          )
        NNBase.__init__(self, param_dims)

        #self.sparams.L = wv.copy() # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)
        self.params.G = random_weight_matrix(*self.params.G.shape)
        self.outputsize = dims[3]

    def forward_pass(self, x):
        z1 = self.params.W.dot(x) + self.params.b1
        h1 = tanh(z1)
        z2 = np.dot(self.params.U, h1) + self.params.b2
        h2 = tanh(z2)
        z3 = np.dot(self.params.G, h2) + self.params.b3
        y_hat = softmax(z3)
        return y_hat


    def _acc_grads(self, x, label):
        """
        Accumulate gradients from a training example.
        """
        #import ipdb; ipdb.set_trace()
        ##
        # Forward propagation
        z1 = self.params.W.dot(x) + self.params.b1
        h1 = tanh(z1)
        z2 = np.dot(self.params.U, h1) + self.params.b2
        h2 = tanh(z2)
        z3 = np.dot(self.params.G, h2) + self.params.b3
        y_hat = softmax(z3)
        
        y = make_onehot(label, self.outputsize)
        d3 = y_hat - y
        self.grads.b3 += d3
        self.grads.G += np.outer(d3, h2) + self.lreg * self.params.G

        d2 = np.dot(self.params.G.T, d3) * tanhd(z2)

        self.grads.b2 += d2
        self.grads.U += np.outer(d2, h1) + self.lreg * self.params.U
        d1 = np.dot(self.params.U.T, d2) * tanhd(z1) 
        
        self.grads.W += np.outer(d1, x) + self.lreg * self.params.W
        self.grads.b1 += d1


    def compute_loss_full(self, X, y):
        #import ipdb; ipdb.set_trace()
        loss = 0
        for idx in xrange(X.shape[0]):
            loss += self.compute_loss(X[idx], y[idx])
        Jreg = (self.lreg / 2.0) * (np.sum(self.params.W**2.0) + 
                np.sum(self.params.U**2.0) + np.sum(self.params.G**2.))
        return loss + Jreg


    def compute_loss(self, x, label):
        """
        Compute the cost function for a single example.
        """
        # Forward propagation
        y_hat = self.forward_pass(x)
        return -np.log(y_hat[label])
    
    def predict_proba(self, X):
        # we can guess numb of class by params.b
        return np.apply_along_axis(self.predict_proba_single, 1, X)

    def predict_proba_single(self, x):
        """
        Predict class probabilities.
        """
        p = self.forward_pass(x)
        return p
    
    def predict(self, X):
        return np.apply_along_axis(self.predict_single, 1, X)

    def predict_single(self, x):
        """Predict most likely class."""
        P = self.predict_proba_single(x)
        return np.argmax(P)
    
    def get_weights(self):
        W1 = np.hstack([self.params.W, self.params.b1.reshape(-1, 1)])
        W2 = np.hstack([self.params.U, self.params.b2.reshape(-1, 1)])
        W3 = np.hstack([self.params.G, self.params.b3.reshape(-1, 1)])

        return (W1, W2, W3)
