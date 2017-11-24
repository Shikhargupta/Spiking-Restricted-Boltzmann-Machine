from utils.preprocessors import BasicPreprocessor
from utils.readers import CsvReader
import numpy as np
import re
import matplotlib.pyplot as plt
from classifiers.misc import minibatch, fullbatch
import json


class Clfpipeline(object):
    ''' Abastract class which given  the data input, produces performance 
    metrics, and outputs weight and topology of the model. 
    It is asembled of the following components:
        - Reader
        - Preprocessor
        - Classifier
    '''

    def __init__(self, reader, classifier=None, PreProc=BasicPreprocessor,
            load_now=True):
        
        ''' takes a json arg, extracts and builds parameters 
            reader - data reader object
            classifier classifier object
            preProc - Preprocessor class
            load_now - # if true will laod and prepare now, or else
                        you need to call pipe.prepare_data() later
        '''
        
        # TODO right now read from provided classifier later, build it 
        # up from configuration
        
        self.classifier = classifier
        self.reader = reader
        self.splits = [0.8, 0.1, 0.1]
        self.PreProc = PreProc
        if load_now:
            self.prepare_data()

    def set_classifier(self, classifier):
        self.classifier = classifier

    def prepare_data(self):
        self.X, self.y = self.reader.load_data()

        def hook(X, y): 
            #needed to substract 1 from labels
            return (X, y - min(y))
        
        self.preprocessor = self.PreProc(self.X, self.y, hook)
        self.preprocessor.preprocess_data()
        #np.savetxt('output/iris_data.csv', self.preprocessor.X, delimiter=',', fmt='%.8f') 
        #np.savetxt('output/iris_target.csv', self.preprocessor.y, delimiter=',', fmt='%d') 
        (self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, 
                self.y_test) = self.preprocessor.get_splits(*self.splits)

        print self.X_test.shape, self.y_test.shape
        
    
    def train(self, **params):
        # reconstruct parameters, TODO later you get it from conf file
        # params for training a classifier or autoencoder
        
        #costevery = params['costevery']
        n_train = self.X_train.shape[0]
        #nepoch = params['nepoch']
        #acc_batch = params['acc_batch'] 
        #opt = params['opt']
        #tolerance = params['tolerance']
        #loss_metric = params['loss_metric']
        
        if 'batchsize' in params:
            # the batchsize is an int > 0 if we pass mini batches, it's -1 if 
            # we pass full batches. we need to constract idxiter object which
            # is a python generator for generating sample sequences to train.
            if params['batchsize'] == -1:
                params['idxiter'] = fullbatch(n_train, params['nepoch'])
            else:
                params['idxiter'] = minibatch(n_train, params['batchsize'], 
                        params['nepoch'])
            params.pop('batchsize')
            params.pop('nepoch')

        # loss metric is for pipeline to use only at this point
        loss_metric = params['loss_metric']
        params.pop('loss_metric')

        self.curve = self.classifier.train_sgd(self.X_train, self.y_train,  
                devX = self.X_dev, devy = self.y_dev, **params)
        #counts, costs, costdevs  = zip(*curve)
        
        y_hat_train = self.classifier.predict(self.X_train)
        y_hat_dev = self.classifier.predict(self.X_dev)
        y_hat_test = self.classifier.predict(self.X_test)
        
        predictions  = {'TRAIN':(self.y_train, y_hat_train),
                'VALID':(self.y_dev, y_hat_dev),
                'TEST':(self.y_test, y_hat_test)} 
        
        self.losses = {} 

        if loss_metric == 'accuracy':
            self.calc_losses_accuracy(predictions)
        elif loss_metric == 'MSE':
            self.calc_losses_mse(predictions)
        else:
            raise Exception('Unknown loss function')
    

    def calc_losses_accuracy(self, predictions):
        for set_name, (y, y_hat) in predictions.items():
            accuracy = self.calc_accuracy(y, y_hat)
            print 'Accuracy on %s %.4f' % (set_name, accuracy)
            self.losses[set_name] = accuracy

    def calc_losses_mse(self, predictions):
        for set_name, (y, y_hat) in predictions.items():
            mse = self.calc_mse(y, y_hat)
            print 'MSE on %s %.4f' % (set_name, mse)
            self.losses[set_name] = mse

    def calc_accuracy(self, y, y_hat):
        return np.count_nonzero(y == y_hat) / float(len(y_hat))

    def calc_mse(self, y, y_hat):
        print y.shape
        print y_hat.shape
        return np.mean(np.mean((y-y_hat) ** 2, axis=-1))

    def plot(self):
        # plot last training run
        # TODO can be moved outside of this module
        counts, costs, costdevs  = zip(*self.curve)

        plt.figure(figsize=(6,4))
        plt.plot(5*np.array(counts), costs, color='b', marker='o', linestyle='-', label=r"train_loss")
        plt.plot(5*np.array(counts), costdevs, color='g', marker='o', linestyle='-', label=r"validation_loss")

        #plt.title(r"Learning Curve ($\lambda=0.001$, costevery=5000)")
        plt.xlabel("SGD Iterations"); plt.ylabel(r"Average $J(\theta)$"); 
        plt.ylim(ymin=0, ymax=max(1.1*max(costs),3*min(costs)));
        plt.legend()

    
    def save_weights_old(self, filename):
        weights =  self.classifier.get_weights()
        dims = self.classifier.dims
        with open(filename, 'w') as f:
            f.write(' '.join(map(str, dims)) + '\n')   # write dimensions
            for w in weights:
                f.write('\n')
                f.write(' '.join(map(str, w.shape)) + '\n' )
                # now start writing matrix
                for r in w:
                    f.write(' '.join(map(str, r)) + '\n')

                #matr_str = re.sub('[\[\]]', '', np.array_str(w))
                #f.write(matr_str)
    
    def save_weights(self, algo_name, dataset_name, confs, folder='output/'):
        # TODO this part was coded fast, need to refarcotr it
        weights =  self.classifier.get_weights()
        dims = self.classifier.dims
        # save preprocessed weights as well
        dt_fn = '%s/data.csv' % (folder)
        lb_fn = '%s/labels.csv' % (folder)
        
        # if needed to save preprocessed data
        #np.savetxt(dt_fn, self.preprocessor.X, fmt='%.10f')
        #np.savetxt(lb_fn, self.preprocessor.y, fmt='%d')


        # write description
        desc_filename = (folder + algo_name + '_' + 
                dataset_name + '_' + 'DESC' + '.txt')
        print 'descript filename', desc_filename
        with open(desc_filename, 'w') as f:
            msgs = []
            msgs.append ('# %s for %s DATASET' % (algo_name, dataset_name))
            msgs.append('# SPLITS(train, val, test) %.3f, %.3f, %.3f' % 
                    tuple(self.splits))

            #msgs.append('# Losses(train, val, test: %.3f, %.3f, %.3f' % 
                    #tuple(self.losses))
            # TODO add losses
            print msgs
            f.write('\n'.join(msgs) + '\n')
            for d in dims:
                f.write(str(d) + '\n')

            f.write('# Internal parameters of network and training \n')
            jsonstr = json.dumps(confs)
            f.write(jsonstr)
        
        for i, w in enumerate(weights):
            wght_fn = '%s%s_%s_weights_%d.txt' % (folder, algo_name, 
                    dataset_name, i + 1)
            print 'weight filename', wght_fn
            with open(wght_fn, 'w') as f:
                #import ipdb; ipdb.set_trace()
                f.write('\n'.join(map(str, w[:, -1])))
                f.write('\n')
                for col in range(0, w.shape[1] - 1):
                    f.write('\n'.join(map(str, w[:, col])))
                    f.write('\n')

    def get_weights(self):
        return self.classifier.get_weights()

