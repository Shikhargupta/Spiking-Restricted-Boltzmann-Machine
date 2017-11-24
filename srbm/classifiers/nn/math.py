from numpy import *

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1. - sigmoid(x))

def softmax(x):
    xt = exp(x - max(x))
    return xt / sum(xt)

def tanh(x):
    return 2 * sigmoid(2 * x) - 1.

def make_onehot(i, n):
    y = zeros(n)
    y[i] = 1
    return y

def tanhd(x):
    # deriative of tan function
    #return 4 * sigmoid(2.0 * x) * (1.0 - sigmoid(2.0 * x))
    return 1. - tanh(x) ** 2


class MultinomialSampler(object):
    """
    Fast (O(log n)) sampling from a discrete probability
    distribution, with O(n) set-up time.
    """

    def __init__(self, p, verbose=False):
        n = len(p)
        p = p.astype(float) / sum(p)
        self._cdf = cumsum(p)

    def sample(self, k=1):
        rs = random.random(k)
        # binary search to get indices
        return searchsorted(self._cdf, rs)

    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def reconstruct_p(self):
        """
        Return the original probability vector.
        Helpful for debugging.
        """
        n = len(self._cdf)
        p = zeros(n)
        p[0] = self._cdf[0]
        p[1:] = (self._cdf[1:] - self._cdf[:-1])
        return p


def multinomial_sample(p):
    """
    Wrapper to generate a single sample,
    using the above class.
    """
    return MultinomialSampler(p).sample(1)[0]
