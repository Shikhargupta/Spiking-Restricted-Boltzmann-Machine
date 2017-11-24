from collections import namedtuple
import numpy as np
from uvnn.utils.readers import CsvReader
import pandas as pd

Param = namedtuple('Parameters', 
        ['eta', 'thresh_eta', 'numspikes', 'timespan', 'tau', 'thr',
            'inp_scale', 't_refrac', 'stdp_lag', 'min_thr', 'plot_things', 
            'axon_delay', 't_gap'])
Spike = namedtuple('Spike',
        ['time', 'layer', 'address'])

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def data_to_spike(x, numspikes, timespan):
    ''' return pairs of spike_address, time during time [0 - timespan]'''
    probs = x / float(sum(x))
    spikes = np.random.choice(range(len(x)), numspikes, True, probs)
    times = np.sort(np.random.rand(numspikes) * timespan)
    return zip(spikes, times)

def load_data(args, logger):
    ''' Load data from file specified by args, or load default dataset '''

    # y might be None as the algorithm is unsupervised 
    if args.input_file is None: 
        # by default we take kaggle 28x28 dataset
        if args.num_train > 10: #TODO remove later not needed
            csv_reader = CsvReader(fn='../../input/kaggle_mnist/train.csv',
                    has_header=True, label_pos=0)
        else:
            csv_reader = CsvReader(fn='../../input/kaggle_mnist/train.csv',
                    has_header=True, label_pos=0)
        X, y = csv_reader.load_data()
        y = y.astype(int)
    else:
        csv_reader = CsvReader(fn=args.input_file, has_header=True,label_pos=0)
        X, y = csv_reader.load_data()
        # for i in range(len(y)):
        #     y[i] = int(y[i])
        y = y.astype(int)

    logger.info('Data Loaded, shape is %s' % (X.shape,))
    if y is None:
        logger.info('Labels weren"t Loaded!, can only be trained unsupervised way')
    else:
        logger.info('Labels Loadded, shape is %s' %(y.shape,))

    return (X, y)

def prepare_dataset(X, y, args):
    ''' Do basic preprocessing, split the dataset into test and train sets ''' 

    # X = (X - np.min(X)) / float(np.max(X) - np.min(X)) # normalize 0 .. 1
    
    if args.shuffle:
        order = np.array(range(X.shape[0]))
        np.random.shuffle(order)
        X = X[order]
        y = y[order]

    num_test = args.num_test
    num_train = args.num_train

    # put away some part for testing purposes if desired
    X_test = X[:num_test]
    X_train = X[num_test: num_test + num_train]
    if y is not None: 
        y_test = y[:num_test]
        y_train = y[num_test: num_test + num_train]
    else:
        y_test = y_train = None

    return X_train, y_train, X_test, y_test

def normalize(x):
    ''' input  - x numpy array
        output normalized between 0 and 1
    ''' 
    # normalize between 0 and 1
    # normalized = (x-np.min(x))/float(np.max(x)-np.min(x)) 
    return normalized
