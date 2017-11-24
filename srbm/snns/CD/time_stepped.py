import numpy as np
import copy
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from Queue import PriorityQueue
from common import data_to_spike, prepare_dataset#, wrapText
from common import Param, Spike, str2bool, data_to_spike, prepare_dataset
from uvnn.utils.images import show_images
import pandas as pd

class SRBM_TS(object):
    ''' Spiking RBM, Time-stepped implementation '''
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def set_data(self, X, y):
        self.X = X
        self.y = y

    def run_network(self, X, y, args, phase='TRAIN', W_init=None, examples_seen = 0, 
            accuracies = [], heatmaps = []):
        ''' phase is either TRAIN or EVAL '''
        logger = self.logger
        batch_size = args.batch_size

        num_train = X.shape[0]
        num_batches = ((num_train - 1) / batch_size) + 1

        vis_size = args.visible_size
        hid_size = args.hidden_size
        if args.train_supervised:
            vis_size += args.num_classes


        layer_sizes = [vis_size, hid_size, vis_size, hid_size]
        
        if phase == 'EVAL':
            # evaluate accuracy, using uncoupling pretrained network see page 29 
            logger.info('Running evaluation')
            layer_sizes.append(args.num_classes)
        
        if W_init is None:
            # use random weight
            if args.load_weights is not None:
                W = np.load(args.load_weights)
            else:
                # W = np.random.random((vis_size, hid_size))
                W = np.random.uniform(low=0,high=0.1,size=(vis_size,hid_size))  ####### Weight initialization changed to 0-0.1 #######

        if phase == 'EVAL':
            if W_init is None:
                logger.error('Weights should be provided to evaluate ' )

            # uncouple W and W_label
            W = W_init[:vis_size,:]
            W_label = W_init[vis_size:,:]
        
        for epoch in range(args.num_epoch):
            #logger.info('running Epoch #%d' %(epoch, )) 
            # initialize params
            membranes = []
            refrac_end = []
            last_spiked = []
            firings = []
            noises = []
            noise_uniform = args.noise_uniform
            for lsz in layer_sizes:
                membranes.append(np.zeros((batch_size, lsz)))
                last_spiked.append(np.zeros((batch_size, lsz)))
                last_spiked[-1].fill(-10000) # not to cause first spikes
                refrac_end.append(np.zeros((batch_size, lsz)))
                refrac_end[-1].fill(-10000) # not to cause first spikes
                firings.append(np.zeros((batch_size, lsz)))
                noises.append(np.zeros((batch_size, lsz)))
           
            cur_time = 0
            classified_correctly = 0 # used when evaluating
            heatmap = np.zeros((args.num_classes, args.num_classes), dtype=int)
            for batch in range(num_batches):
                #if not evaluate:
                #    logger.info('Processing batch from #%d' %(batch * batch_size, ))
                label_firing_count = np.zeros((batch_size, args.num_classes)) # for classification only not for training
                total_input_spikes = 0
                total_output_spikes = 0
                for t in np.arange(cur_time, cur_time + args.timespan, args.dt): 
                    # get batch spikes
                    l = batch * batch_size
                    r = l + batch_size
                    in_current = X[l:r,:]
                    if y is not None:
                        lab_current = y[l:r]
                 
                    if args.train_supervised:
                        in_current_labels = np.zeros((batch_size, args.num_classes))
                        in_current_labels[range(batch_size), lab_current] = 0.9
                        in_current = np.hstack((in_current, in_current_labels))

                        
                        
                
                    rand_nums = np.random.random((batch_size, vis_size))
                    rand_noise = np.random.random((batch_size, vis_size)) * noise_uniform
                    noise_uniform *= args.noise_decay
                    firings[0] =  (rand_nums  < in_current + rand_noise)
                    
                    #print np.count_nonzero(firings[0])



                ############################################## Limiting total input spikes ###############################
                    total_input_spikes += np.count_nonzero(firings[0])
                    # print total_input_spikes
                    if(total_input_spikes>args.numspikes):
                        break
                ##########################################################################################################        


                   # import ipdb; ipdb.set_trace()
                    last_spiked[0][firings[0]] = t
                    
                    for pop in range(1, len(layer_sizes)):
                        # decay membrane
                        #import ipdb; ipdb.set_trace()


                        ## Pot depreciation of each layer - linear or non linear according to the input arg given
                        if args.linear_decay is None or (args.linear_decay_only_in_eval 
                                and phase == 'TRAIN'):
                            membranes[pop]*= np.exp(-args.dt / args.tau)
                        else:
                            membranes[pop]-= args.linear_decay
                            membranes[pop][membranes[pop] < 0] = 0

                        

                        #if membranes[pop].any():
                        #    import ipdb; ipdb.set_trace()
                        #membranes[pop] = membranes[pop] * np.exp(-args.dt / args.tau)
                         
                        # add impulses
                        if pop == 1 or pop == 3:
                            membranes[pop] += (t > refrac_end[pop]) *  np.dot(firings[pop - 1], W)
                        elif pop == 2:
                            # model visible layer
                            membranes[pop] += (t > refrac_end[pop]) * np.dot(firings[pop - 1], W.T)

                        #flag!!!!    
                        elif pop == 4:
                            # this is a label layer directly connected to
                            # the hidden layer
                            membranes[pop] += (t > refrac_end[pop]) * np.dot(firings[1], W_label.T)
                        

                        # get firings if greater than threshold
                        firings[pop] = membranes[pop] > args.thr
                        #if firings[pop].any():
                        #    print np.average(firings[pop]

                        ## update potential of firing neurons and send them to refractory period
                        membranes[pop][firings[pop]] = 0
                        refrac_end[pop][firings[pop]] = t + args.t_refrac
                        last_spiked[pop][firings[pop]] = t

                        
                        if pop == 4:
                            # add label firing count 
                            label_firing_count += firings[pop]
                        
                    # now learn if not evaluating

################################################################## 22/06/2017 ########################################################################
############################################################## Added Neftci's curve of stdp update ###################################################
                    total_output_spikes += np.count_nonzero(firings[1])
                    # print total_output_spikes
                    # stdp but with constant factor
                    ## flag!!!!
                    # print np.shape(last_spiked[0])
                    if phase == 'TRAIN' and args.enable_update:
                        dWp = dWn = np.zeros((vis_size, hid_size))

                        if args.stdp_curve == "NEIL":

                            if np.any(firings[1]):
                                xx = (last_spiked[0] > t - args.stdp_lag) #* (last_spiked[0] < t-1e-9)
                                dWp = args.eta * (np.dot(xx.T, firings[1]))
                            if np.any(firings[3]):
                                xx = (last_spiked[2] > t - args.stdp_lag)
                                dWn = args.eta * (np.dot(xx.T, firings[3]))

                        elif args.stdp_curve == "NEFTCI":
                            tau_nef = args.tau_stdp
                            curr_times = []
                            curr_times.append(t*np.ones(np.shape(last_spiked[0])))
                            curr_times.append(t*np.ones(np.shape(last_spiked[1])))
                            curr_times.append(t*np.ones(np.shape(last_spiked[2])))
                            curr_times.append(t*np.ones(np.shape(last_spiked[3])))
                            # b1 = t*np.ones(np.shape(last_spiked[2]))

                            if np.any(firings[1]):

                                xx = (last_spiked[0] > t - args.stdp_lag) #* (last_spiked[0] < t-1e-9)
                                time_diff = last_spiked[0] - curr_times[0]

                                time_diff[:] = [x /tau_nef for x in time_diff]                                

                                exp_time_diff = np.exp(-np.abs(time_diff))
                                temp = xx*exp_time_diff
                                dWp = dWp + args.eta * (np.dot(temp.T, firings[1]))

                            if np.any(firings[0]):

                                xx = (last_spiked[1] > t - args.stdp_lag) #* (last_spiked[0] < t-1e-9)

                                time_diff = last_spiked[1] - curr_times[1]
                                time_diff[:] = [x / tau_nef for x in time_diff]

                                exp_time_diff = np.exp(-np.abs(time_diff))
                                temp = xx*exp_time_diff
                                
                                dWp_temp = args.eta * (np.dot(temp.T, firings[0]))
                                dWp = dWp + dWp_temp.T    

                            if np.any(firings[3]):

                                xx = (last_spiked[2] > t - args.stdp_lag) #* (last_spiked[0] < t-1e-9)
                                time_diff = last_spiked[2] - curr_times[2]

                                time_diff[:] = [x / tau_nef for x in time_diff]
                                
                                exp_time_diff = np.exp(-np.abs(time_diff))
                                temp = xx*exp_time_diff
                                dWn = dWn + args.eta * (np.dot(temp.T, firings[3]))

                            if np.any(firings[2]):

                                xx = (last_spiked[3] > t - args.stdp_lag) #* (last_spiked[0] < t-1e-9)
                                time_diff = last_spiked[3] - curr_times[3]

                                time_diff[:] = [x / tau_nef for x in time_diff]
                                

                                exp_time_diff = np.exp(-np.abs(time_diff))
                                temp = xx*exp_time_diff
                                dWn_temp = args.eta * (np.dot(temp.T, firings[2]))
                                dWn = dWn + dWn_temp.T
                        
                        dW = (dWp - dWn) / batch_size
                        #if dW.any():
                        #    import ipdb; ipdb.set_trace()
                        W += dW
###################################################################################################################################################
###################################################################################################################################################

                cur_time = cur_time + args.timespan + args.t_gap
                for pop in range(1, len(layer_sizes)):
                    membranes[pop] *= np.exp(-args.t_gap / args.tau)
                if args.train_supervised:
                    examples_seen = (batch + 1) * batch_size
                    if examples_seen / args.test_every > (examples_seen - batch_size) / args.test_every:

                        logger.info('Processed #%d examples' %(examples_seen, ))
                        logger.info('Noise is %.3f' %(noise_uniform,))
                        # time to evaluate network
                        newargs = copy.copy(args)
                        newargs.batch_size = 1
                        newargs.train_supervised = False
                        self.run_network(self.X_test, self.y_test, newargs, 
                                phase='EVAL', W_init=W, examples_seen=examples_seen, 
                                accuracies=accuracies, heatmaps=heatmaps)
                
                if phase == 'EVAL':
                    #print label_firing_count[0]
                    #print 'correct is ', lab_current[0]
                    y_hats = np.argmax(label_firing_count, axis=1)
                    classified_correctly += np.count_nonzero(y_hats == lab_current)
                    heatmap[y_hats, lab_current] += 1

        if phase == 'EVAL':
            logger.info('Correctly classified %d out ouf %d' %(classified_correctly, len(y)))
            acc = classified_correctly / float (len(y))
            print acc
            accuracies.append((examples_seen, acc))
            heatmaps.append(heatmap)
            plt.close()
            #plt.figure(figsize=(5, 5))
            print 'minmax', np.min(W), np.max(W), np.average(W)
            #print heatmap
            if args.plot_curve:
                show_images(W.T, 28, 28)
                plt.show(block=False)
        return {}, W


    def train_network(self):
        
        self.X_train, self.y_train, self.X_test, self.y_test =  prepare_dataset(self.X, self.y, self.args)
        self.logger.info('Dataset prepared, Ttrain, Test splits ready to use')
        accuracies = []
        heatmaps = []
        hist, W = self.run_network(self.X_train, self.y_train, self.args, phase='TRAIN', 
                accuracies=accuracies, heatmaps=heatmaps)
        #import pickle
        #pickle.dump( accuracies, open( "accs.p", "wb" ) )
        last_heatmap = heatmaps[-1]

        ######################################### 20/06/2017 ###########################################################################
        #################### Creating a data frame for final heatmap (confusion matrix) and saving it as a csv file ####################

        columns = range(self.args.num_classes)

        conf_csv = pd.DataFrame(columns=columns)
        for index in range(self.args.num_classes):
            conf_csv.loc[index] = last_heatmap[index]

        conf_csv.to_csv("statistical_data/confusion_matrix.csv")
        
        #################################################################################################################################

        if self.args.plot_curve:
            fig = plt.figure()
            #fig.suptitle(3*'\n'.join(str(self.args).split(',')), fontsize=8)
            ax1 = fig.add_subplot(221)
            ax1.set_xlabel('Samples seen')
            ax1.set_ylabel('Accuracies')
            examples_seen, accs = zip(*accuracies)
            ax1.plot(examples_seen, accs)
            
            ax2 = fig.add_subplot(222)
            sns.heatmap(last_heatmap, annot=True, fmt='d', ax=ax2)
            ax2.set_xlabel('True values')
            ax2.set_ylabel('Predicted values')

            ax3 = fig.add_subplot(223)
            #ax2.text(0, 0, 'aaa')
            ax3.set_xlabel('Parameters')
            an = ax3.annotate(str(self.args), fontsize=10, xy=(0.1, 1), ha='left', va='top', xytext=(0, -6),
                            xycoords='axes fraction', textcoords='offset points')
            #wrapText(an)
            #ax2.text(0.5, 0.5,str(self.args), horizontalalignment='center',
            #        verticalalignment='center', transform=ax2.transAxes, fontsize=8)
            plt.show(block=True)
        return hist, W, accuracies
