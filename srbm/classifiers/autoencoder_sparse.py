# TODO move to outside classifier package or change the name of "classifiers"
from nn.base import NNBase
from nn.math import softmax, make_onehot, tanh, tanhd, sigmoid_grad, sigmoid
from misc import random_weight_matrix
import numpy as np


def KLD(ro_hat, ro):
        "returns KL divergence between vector ro_hat and a number ro"
        #import ipdb; ipdb.set_trace()
        ret = 0
        for i in xrange(len(ro_hat)):
            ret = ret + (ro * np.log(ro / ro_hat[i]) + 
                    (1 - ro) * np.log((1 - ro) / (1 - ro_hat[i])))
        return ret


class AutoEncoderSparse(NNBase):
    def __init__(self, dims=[100, 5, 100],
                 reg=0.1, alpha=0.001, ro = 0.05,
                 rseed=10, beta=0.2):
        """
        Set up autoencoder: parameters, hyperparameters
        """
        ##
        # Store hyperparameters
        self.lreg = reg # regularization
        self.alpha = alpha # default learning rate
        self.dims = dims # todo move to superclass
        self.ro = ro # ro sparsity parameter
        self.beta = beta  # sparsity penalty 


        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        NNBase.__init__(self, param_dims)

        #self.sparams.L = wv.copy() # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)
        self.outputsize = dims[2]

    def forward_pass(self, x):
        " Compute the final and the hidden layer "
        z1 = self.params.W.dot(x) + self.params.b1
        h = sigmoid(z1)
        z2 = np.dot(self.params.U, h) + self.params.b2
        return (z2, h)

    def _acc_grads(self, x, y):
        self._acc_grads_batch([x], [y])


    def _acc_grads_batch(self, X, Y):
        """
        Accumulate gradients from a training examples,
        X matrix, Y vector of targets
        TODO hidden layer average activation is done separately 
        can be rewritten to be twice as fast
        """
        # Frist compute average activation for examples
        ro_hat = np.zeros_like(self.params.b1)
        for i in range(len(Y)):
            x = X[i]
            _, h = self.forward_pass(x)
            ro_hat += h
        ro_hat /= float(len(Y))


        ##
        # Forward propagation
        for i in range(len(Y)):
            x = X[i]
            y = Y[i]
            z1 = self.params.W.dot(x) + self.params.b1
            h = sigmoid(z1)
            z2 = np.dot(self.params.U, h) + self.params.b2
            y_hat = z2
            
            d2 = (y_hat - y) 
            #d2 *= (1./len(y))
            self.grads.b2 += d2
            self.grads.U += np.outer(d2, h) + self.lreg * self.params.U
            
            # incorporate kld gradient into d1
            kl_grad = self.beta * (- self.ro / ro_hat + 
                    (1. - self.ro) / (1 - ro_hat))
            d1 = (np.dot(self.params.U.T, d2) + kl_grad) * sigmoid_grad(z1)
            
            self.grads.W += np.outer(d1, x) + self.lreg * self.params.W
            self.grads.b1 += d1

    def compute_loss_full(self, X, Y):
        #import ipdb; ipdb.set_trace()
        """ Compute the full loss of sparse autoencoder, scoring is MSE,
          regularization and sparsity penalty term is add up """
        full_loss = 0
        ro_hat = np.zeros_like(self.params.b1) #average activation
        for idx in xrange(X.shape[0]):
            x = X[idx]
            y = Y[idx]
            y_hat, hidd = self.forward_pass(x)
            loss = np.sum((y_hat - y) ** 2)
            full_loss += loss
            ro_hat += hidd

        #import ipdb; ipdb.set_trace()
        Jreg = (self.lreg / 2.0) * (np.sum(self.params.W**2.0) + 
                np.sum(self.params.U**2.0))

        ro_hat /= len(Y)
        
        Jsparsity = KLD(ro_hat, self.ro)
        #print 'full_loss', full_loss * 0.5
        print 'Jsparsity', Jsparsity
        # sparsity term loss
        return 0.5 * full_loss + self.beta * Jsparsity + Jreg

    
    def predict(self, X):
        return np.apply_along_axis(self.predict_single, 1, X)

    def predict_single(self, x):
        y_hat, hidd = self.forward_pass(x)
        return y_hat
    def predict_hidden_single(self, x):
        y_hat, hidd = self.forward_pass(x)
        return hidd

    def predict_hidden(self, X):
        # get hidden layers instead of predicitons, used for stacked training
        return np.apply_along_axis(self.predict_hidden_single, 1, X)
    
    def get_weights(self):
        W1 = np.hstack([self.params.W, self.params.b1.reshape(-1, 1)])
        W2 = np.hstack([self.params.U, self.params.b2.reshape(-1, 1)])
        return (W1, W2)
