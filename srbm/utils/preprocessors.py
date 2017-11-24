''' Module for Data preprocessors, take X and y row data as an argument and 
preprocesses it and returns train, test validation splits ''' 

import numpy as np
import pandas as pd

class BasicPreprocessor(object):
    def __init__(self, X, y, hook=None):
        ''' sometimes it's necessary to do some simple operations  before 
        preprocessing, i.e. labels are counted from 1 instead from 0 hook 
        should be the function which takes X, y and returns X, y after applying 
        this simple operation.
        '''
        self.X = X
        self.y = y
        self.hook = hook       
        self.num_samples, num_features = self.X.shape

    def preprocess_data(self):
        ''' Basic preprocessing normalizes the data, substracts mean
        and divides by sd '''
        self.y = np.array(self.y).flatten().astype(int)
        if self.hook is not None:
            self.X, self.y = self.hook(self.X, self.y)
        
        mean = np.mean(self.X, axis=0)
        self.X = self.X - mean
        sd = np.std(self.X, axis=0)
        nonzero = sd > 0
        
        self.X[:, nonzero] /= sd[nonzero]
    
    def get_splits(self, frac_train, frac_dev, frac_test, shuffle=True):
        ''' returns train, validation and test splits
            - frac_train, frac_dev, frac_test : fraction of test splits,
            for example 0.8, 0.1, 0.1
            - shuffle: whether or not to shuffle rows before splitting
        '''
        
        # split the data 
        num_train = int(frac_train * self.num_samples)
        num_dev = int(frac_dev * self.num_samples)
        num_test = int(frac_test * self.num_samples)

        if shuffle: 
            ind = range(self.num_samples)
            np.random.shuffle(ind)
            self.X = self.X[ind, :]
            self.y = self.y[ind]
        
        X_train = self.X[:num_train, :]
        y_train = self.y[:num_train]
        print 'X_train shape', X_train.shape

        X_dev = self.X[num_train:num_train + num_dev,:]
        y_dev = self.y[num_train:num_train + num_dev]
        
        print 'X_dev shape', X_dev.shape

        X_test = self.X[num_train + num_dev:num_train + num_dev + num_test,:]
        y_test = self.y[num_train+ num_dev:num_train + num_dev + num_test]
        
        print 'X_test shape', X_test.shape
        return (X_train, y_train, X_dev, y_dev, X_test, y_test)

class AutoEncoderPP(BasicPreprocessor):
    def __init__(self, X, y, hook=None):
        BasicPreprocessor.__init__(self, X, y, hook)

    def preprocess_data(self):
        #self.X = self.X / 255.
        mean = np.mean(self.X, axis=0)
        self.X = self.X - mean
        #sd = np.std(self.X, axis=0)
        #nonzero = sd > 0
        
        #self.X[:, nonzero] /= sd[nonzero]

        self.y = self.X

class ColorImagePP(BasicPreprocessor):
    # Class name says the 
    def __init__(self, X, y, hook=None):
        BasicPreprocessor.__init__(self, X, y, hook)

    def preprocess_data(self):
        self.X = self.X / 255.
        #mean = np.mean(self.X, axis=0)
        #self.X = self.X - 0.5
        #sd = np.std(self.X, axis=0)
        #nonzero = sd > 0
        
        #self.X[:, nonzero] /= sd[nonzero]

        self.y = self.X

class MnistPP(BasicPreprocessor):
    def __init__(self, X, y, hook=None):
        BasicPreprocessor.__init__(self, X, y, hook)

    def preprocess_data(self):
        self.X = self.X / 255.
        #self.X = self.X - 0.5
        self.y = self.X


