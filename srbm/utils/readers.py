''' Module for Data readers, they take the file name etc, as an argument
and return arrays X and y '''

import numpy as np
import pandas as pd

class CsvReader(object):
    def __init__(self, fn, has_header, label_pos=None, fn_labels=None, sep=','):
        ''' prepare reader to load the data
            - fn: name of the file to read
            - fn_labels: name of labels filename, if labels are in the
              same file it can be set to None
            - label_pos: label position in the csv file, if label is in the
              same file.
            - has_header: weather or not the file has header
            - sep: separator
        '''
        self.fn = fn
        self.fn_labels = fn_labels
        self.label_pos = label_pos
        self.header = 'infer' if has_header else None 
        self.sep = sep

    def load_data(self):
        ''' return Data in the form of X and y '''
        fulltrain = pd.read_csv(self.fn, sep=self.sep, 
                header=self.header)
        fulltrain = fulltrain.as_matrix() # convert to numpy array
        n_samples, n_features = fulltrain.shape
        
        if self.fn_labels is None:
            # labels are in the same file
            if self.label_pos is None:
                # We only need have the input in this case, used for
                # autoencoders
                fulltrainX = fulltrain
                fulltrainy = None
            else:
                # we have the label_post. and the targets are in the same file
                features = np.ones(n_features, dtype=np.bool)
                features[self.label_pos] = False # feature holds everything except labels
                fulltrainX = fulltrain[:, features]
                fulltrainy = fulltrain[:, self.label_pos] 
        else:
            labels = pd.read_csv(self.fn_labels, sep=self.sep, 
                    header=self.header)
            fulltrainX = fulltrain
            fulltrainy = labels.as_matrix()
        
        return (fulltrainX, fulltrainy)

class DummyReader(object):
    ''' Useful reader when we already have the data from somewhere else then file'''
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def load_data(self):
        return (self.X, self.y)

