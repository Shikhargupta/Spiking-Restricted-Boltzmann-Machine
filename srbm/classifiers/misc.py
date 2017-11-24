##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    e = sqrt(6) / sqrt(n + m + 1)
    A0 = random.uniform(-e, e, (m, n))
    assert(A0.shape == (m,n))
    return A0



# Different batch strategies

def fullbatch(sample_count, nepoch):
    ''' full batch nepoch times '''
    for i in xrange(nepoch):
        yield range(sample_count)

def minibatch(sample_count, batch_size, epoch):
    ''' return random batch of batch_size for
    epoch * sample_count / batch_size times '''
    for _ in xrange (epoch * sample_count / batch_size):
        yield random.choice(range(sample_count), batch_size, replace=False)

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def vis_images(dataset, classifier, img_n, img_m, n):
    """ Visualize input and output images of autoencoder
    img_n, img_m : image dimensions
    n : number of images to display
    """
    import numpy as np
    mn = np.min(dataset)
    mx = np.max(dataset)
    print mn, mx
    def num_to_col(num):
        return 255. / (mx - mn) * (num + abs(mn))
    vfunc = vectorize(num_to_col)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        mg = vfunc(dataset[i])
        plt.imshow(mg.reshape(img_n, img_m))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        dmg = classifier.predict_single(dataset[i])
        plt.imshow(dmg.reshape(img_n, img_m))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
