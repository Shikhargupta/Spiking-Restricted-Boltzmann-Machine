# General multi layer pereptron for anu number of layers

from nn.base import NNBase
from nn.math import softmax, make_onehot, tanh, tanhd, sigmoid_grad, sigmoid

from misc import random_weight_matrix
import numpy as np

class MLP(NNBase):
    """
    Dummy example, to show how to implement a network.
    This implements softmax regression, trained by SGD.
    """

    def __init__(self, dims=[100, 30, 20, 5],
                 reg=0.1, alpha=0.001,
                 rseed=10, activation='tanh', init_weights=[]):
        """
        Set up classifier: parameters, hyperparameters
        """
        ##
        # Store hyperparameters
        self.lreg = reg # regularization
        self.alpha = alpha # default learning rate
        self.nclass = dims[-1] # number of output classes
        self.dims = dims # todo move to superclass
        self.outputsize = dims[-1]

        
        ## We name the parameters as following 
        # W1, b1, W2, b2, W3, b3 ... 
        param_dims = {}
        for i in range(1, len(dims)):
            w_param = 'W' + str(i)
            b_param = 'b' + str(i)
            param_dims[w_param] = (dims[i], dims[i-1])
            param_dims[b_param] = (dims[i], )

        NNBase.__init__(self, param_dims)

        # set activation function
        if activation =='tanh':
            self.act = tanh
            self.act_grad = tanhd
        elif activation == 'sigmoid':
            self.act = sigmoid
            self.act_grad = sigmoid_grad
        else:
            raise 'Uknown activation function'

        #self.sparams.L = wv.copy() # store own representations
        # init weights
        
        # layers for which init_weights aren't passed are initialized randomly
        for i in range(1, len(self.dims)):
            if i - 1 < len(init_weights):
                # we have the corresponding weights passed for this layer
                cur_weight = init_weights[i-1]
                assert cur_weight.shape == (dims[i], dims[i - 1]), ("passed initial weight dimensions don't match")
            else:
                cur_weight = random_weight_matrix(dims[i], dims[i -1])
            self._set_param('W', i, cur_weight)

    def _set_param(self, param_name, ind, val):
        ''' set parameters
            param name : b or W
            ind : number of parameter W1, b1 (first layer) W2, b2, ..
            val : value to set
        '''
        full_name = param_name + str(ind)
        self.params.__setitem__(full_name, val)

    def _get_param(self, param_name, ind):
        ''' get parameters
            param name : b or W
            ind : number of parameter W1, b1 (first layer) W2, b2, ..
        '''
        full_name = param_name + str(ind)
        return self.params.__getitem__(full_name)

    def _add_grads(self, grad_name, ind, val):
        ''' Add gradient to correspoding param '''
        full_name = grad_name + str(ind)
        old = self.grads.__getitem__(full_name)
        self.grads.__setitem__(full_name, old + val)


    def forward_pass(self, x):
        ''' Forward pass,
        Arguemnt: x input vectore
        Return:(zs, hs) hidden and output activations (hs)
        and inputs to activation function (zs) 

        example:
            input _ dims [100, 30, 20, 5]
            output: hs = [x, h1, h2, h3]
                    zs = [z1, z2, z3]
        '''

        hs = [x]
        zs = []
        h = x
        for i in range(1, len(self.dims)):
            W = self._get_param('W', i)
            b = self._get_param('b', i)
            z = W.dot(h) + b
            zs.append(z)
            # now activation function, if it's a last layer we use softmax
            # else tanh
            if i == len(self.dims) - 1:
                # last layer
                h = softmax(z)
            else:
                h = self.act(z)
            hs.append(h)       
        return (zs, hs)

    def _acc_grads(self, x, label):
        """
        Accumulate gradients from a training example.
        """
        #import ipdb; ipdb.set_trace()
        ##
        # Forward Pass 
        zs, hs = self.forward_pass(x)
        
        y_hat = hs[-1]
        y = make_onehot(label, self.outputsize)
        delta = y_hat - y
        
        cur_h = len(hs) -2 # current h vector index
        cur_z = len(zs) -2 # current z vector index 
        # Backpropagation 
        #import ipdb; ipdb.set_trace()
        for i in range(len(self.dims) - 1, 0, -1):
            self._add_grads('b', i, delta)
            curw = self._get_param('W', i)
            gradw = np.outer(delta, hs[cur_h])
            gradw_reg = self.lreg * curw
            self._add_grads('W', i, gradw + gradw_reg)
            if cur_z >= 0:
                delta = np.dot(curw.T, delta) * self.act_grad(zs[cur_z])
            cur_h -= 1
            cur_z -= 1


    def compute_loss_full(self, X, y):
        ''' compute cross entropy loss for all examples in X
        '''
        loss = 0
        for idx in xrange(X.shape[0]):
            x = X[idx]
            label = y[idx]
            zs, hs = self.forward_pass(x)
            y_hat = hs[-1] # last layer is a final output
            loss -= np.log(y_hat[label])
            
        # now compute Jreg
        Jreg = 0
        for i in range(1, len(self.dims)):
            W = self._get_param('W', i)
            Jreg += np.sum(W**2.)
        Jreg = self.lreg / 2. * Jreg
        return loss + Jreg
    
    def predict_proba(self, X):
        # we can guess numb of class by params.b
        return np.apply_along_axis(self.predict_proba_single, 1, X)

    def predict_proba_single(self, x):
        """
        Predict class probabilities.
        """
        zs, hs = self.forward_pass(x)
        return hs[-1]
    
    def predict(self, X):
        return np.apply_along_axis(self.predict_single, 1, X)

    def predict_single(self, x):
        """Predict most likely class."""
        P = self.predict_proba_single(x)
        return np.argmax(P)
    
    def get_weights(self):
        ''' Get list of weights of each matrix, the baias terms are appended
        as the last column '''
        ret_ws = []
        for i in range(1, len(self.dims)):
            W = self._get_param('W', i)
            b = self._get_param('b', i)
            stacked = np.hstack([W, b.reshape(-1, 1)])
            ret_ws.append(stacked)

        return ret_ws
