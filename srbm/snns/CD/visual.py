
#import initExample

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import collections
import time
from common import Spike, normalize
import matplotlib.pyplot as plt

class DashBoard(object):
    def __init__(self, history, visible_size, hidden_size):
        # history is an array of (time, data) see history definition in evtcd.py
        grid_side = int(np.sqrt(hidden_size - 1)) + 1
        self.w_row = self.w_col = grid_side   # grid for visualizing weights
        
        self.history = history
        self.visible_size = visible_size
        self.hidden_size = hidden_size

        # compute img_sz_viz, by default we take sqrts of sizes
        vis_side = int(np.sqrt(visible_size + 0.1))
        hid_side = int(np.sqrt(hidden_size + 0.1))
        img_sz_vis = (vis_side, vis_side)
        img_sz_hid = (hid_side, hid_side)
        self.layer_sizes = [img_sz_vis, img_sz_hid, img_sz_vis, img_sz_hid]
        self.show_lastn_spike = 400
        self.last_spikes = [collections.deque(
            maxlen=self.show_lastn_spike) for _ in range(5)]
        self.ind = 0 # index in the history array
    

    def plot_reconstr_accuracy(self):
        #import ipdb; ipdb.set_trace()
        cur_img = None  # true distribution of spike train
        recon_img = None# distribution of spikes in reconstraciton layer
        accuracies = []
        min_weights = []
        max_weights = []
        min_weight = 1000
        max_weight = -1000
        for time, events in self.history:
            for event in events: 
                evt_type = event[0]
                if evt_type == 'NEW_SPIKE_TRAIN':
                    if not cur_img is None:
                        # normalize probabilities
                        recon_img = recon_img / float(np.sum(recon_img))
                        cur_img = cur_img / float(np.sum(cur_img))
                        # calculate difference between two distributions
                        # here we just use MSE
                        #import ipdb; ipdb.set_trace()
                        diff = np.sum((cur_img - recon_img) ** 2) / len(cur_img)
                        accuracies.append(diff)

                        # save min max weights as well
                        min_weights.append(min_weight)
                        max_weights.append(max_weight)
                        min_weight = 1000
                        max_weight = -1000
                    cur_img = event[1]
                    recon_img = np.zeros_like(cur_img)
                elif evt_type == 'SPIKE':
                    layer, address = event[1], event[2]
                    if layer == 2:
                        # reconstruction
                        recon_img[address] += 1
                elif evt_type == 'UPDATE_WEIGHTS':
                    weights = event[2]
                    min_weight = np.min(weights)
                    max_weight = np.max(weights)

        #print accuracies 
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(221)
        ax1.plot(accuracies)
        ax1.set_title('accuracy over examples')
        ax2 = fig.add_subplot(222)
        ax2.plot(min_weights)
        ax2.set_title('min_weights')
        ax3 = fig.add_subplot(223)
        ax3.plot(max_weights)
        ax3.set_title('max_weights')
        plt.show()
        

    def run_vis(self):
        app = QtGui.QApplication([])
        
        view = pg.GraphicsView()
        pg.setConfigOptions(antialias=True)
        l = pg.GraphicsLayout(border=(100, 100, 100))
        view.setCentralItem(l)
        view.setWindowTitle('rame')
        view.resize(600, 500)

        view_weights = pg.GraphicsView()
        l_weights_main = pg.GraphicsLayout(border=(100, 100, 100))
        view_weights.setCentralItem(l_weights_main)
        view_weights.show()
        view_weights.setWindowTitle('weights')
        view_weights.resize(400, 400)


        view.show()   #  main view
        # weights layout 
        l_weights = l_weights_main.addLayout(self.w_row, self.w_col, border=(5,0,0))
        l_weights.setContentsMargins(5, 5, 5, 5)
        self.weights_data = None # It will be set by event in the history
        # create weight images, which will be number of hidden layers
        self.weight_imgs = []
        cur_col = 0
        for _ in range(self.hidden_size):
            vb = l_weights.addViewBox(lockAspect=True)
            img = pg.ImageItem(border='w')
            img.setLevels((0, 1))
            self.weight_imgs.append(img)
            vb.addItem(img)
            lr_size = self.layer_sizes[0]
            vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
            cur_col += 1
            if cur_col % self.w_col == 0:
                l_weights.nextRow()

        text = """
        In the spike images row, first image is the input spike train <br>
        Then comes the hidden layer, then visible and hidden again
        """

        l.addLabel(text, col=0, colspan=3)
        time_lab = l.addLabel('rama', color=(255,0,0), size='20pt')
        l.nextRow()
        
        # start spike train image
        ltrain = l.addLayout(colspan=4, border=(10, 0, 0))
        ltrain.setContentsMargins(5, 5, 5, 5)
        #ltrain.addLabel('current Spike train:')
        vb = ltrain.addViewBox(lockAspect=True)
        input_train_img = pg.ImageItem(border='w')
        self.input_train_img = input_train_img
        vb.addItem(input_train_img)
        lr_size = self.layer_sizes[0]
        vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
        l.nextRow()
        # end spike train image

        # start spike images 
        lspikes = l.addLayout(colspan=4, border=(10, 0, 0))
        lspikes.setContentsMargins(5, 5, 5, 5)
        
        self.spike_imgs = []
        self.spike_datas = []
        for lr_size in self.layer_sizes:
            vb = lspikes.addViewBox(lockAspect=True)
            img = pg.ImageItem(border='w')
            self.spike_imgs.append(img)
            vb.addItem(img)
            vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
            self.spike_datas.append(np.zeros(lr_size))
        # end spike images
        
        l.nextRow()
        
        # layer membrane potentials

        lmembranes = l.addLayout(colspan=4, border=(10, 0, 0))
        lmembranes.setContentsMargins(5, 5, 5, 5)


        
        self.membr_imgs = []
        self.membr_datas = []
        for lr_size in self.layer_sizes:
            vb = lmembranes.addViewBox(lockAspect=True)
            img = pg.ImageItem(border='w')
            self.membr_imgs.append(img)
            vb.addItem(img)
            vb.setRange(QtCore.QRectF(0, 0, lr_size[0], lr_size[1]))
            self.membr_datas.append(np.zeros(lr_size))

        # end membrane potentials

        ############# layer for spike imags
        #p1 = l.addPlot(title="Membrane potential1", y=np.random.normal(size=100))
        #p2 = l.addPlot(title='Membrane potential2')
        #p2.plot(y=np.random.normal(size=80))
    
        #vb = l.addViewBox(lockAspect=True)
        #img = pg.ImageItem(border='w')
        #vb.addItem(img)
        #vb.setRange(QtCore.QRectF(0, 0, 100, 100))

        #
        ## Create random image
        #self.img = img
        self.time_lab = time_lab
        self.last_upd_time = 0
        self._update_plots()


        # end image view

        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update_membrane_plot(self, layer, layer_vals):
        img = self.as_image(layer_vals, layer)
        self.membr_datas[layer] = img
        self.membr_imgs[layer].setImage(self.membr_datas[layer])

    def as_image(self, vals, layer):
        ''' converts vector vals to 2D image for visualisation '''
        sz = self.layer_sizes[layer]
        return vals.reshape(sz)

    def update_train_plot(self, vals):
        ''' Plot for Input train image '''
        img = self.as_image(vals, 0)
        self.input_train_img.setImage(img)

    def update_all_weights_plot(self, weights_data):
        for y in range(self.hidden_size):
            self.update_weights_plot(y, weights_data[:, y], False)

    def update_weights_plot(self, column, new_vals, color_delta):
        delta = new_vals - self.weights_data[:, column]
        delta = self.as_image(delta, 0)
        

        # color_delta controls if we wont to color updated regions

        # if positive increased, if negative decreased
        # increasing visualized in red, decreasing in blue
        #import ipdb; ipdb.set_trace()
        #print np.max(delta), np.min(delta)
        self.weights_data[:, column] = new_vals
        x = self.as_image(self.weights_data[:, column], 0)
        normalized = normalize(x) # normalize between 0, 1
        #print 'changed!'
        #if column == 2:
        #    print normalized
        self.weights_imgs_data = np.zeros((28, 28, 3))
        self.weights_imgs_data[:,:,0] = normalized
        self.weights_imgs_data[:,:,1] = normalized
        self.weights_imgs_data[:,:,2] = normalized

        if color_delta: 
            pos_delta = np.nonzero(delta > 0)
            neg_delta = np.nonzero(delta < 0)
            if np.any(neg_delta):
                repl = self.weights_imgs_data[neg_delta]
                repl[:,:] = [0, 0, 1]
                self.weights_imgs_data[neg_delta] = repl
                print 'negative update'
            if np.any(pos_delta):
                repl = self.weights_imgs_data[pos_delta]
                repl[:,:] = [1, 0, 0] # change color to red
                self.weights_imgs_data[pos_delta] = repl
                print 'positive update'

        self.weight_imgs[column].setImage(self.weights_imgs_data)
        #self.weight_imgs[column].setImage(normalized)
    
    def update_spike_plot(self, triplet):
        trip_layer = triplet.layer # original layers were(-1, 0, 1, 2, 3)
        
        # now before adding newspike to correspoding layer trace
        # first delete the last spike on the image
    
        if len(self.last_spikes[trip_layer]) == self.show_lastn_spike:
            oldest_address = self.last_spikes[trip_layer][0].address
            x, y = self.getxy(oldest_address, trip_layer)
            self.spike_datas[trip_layer][x][y] = 0

        # fade everything
        self.spike_datas[trip_layer] *= 0.99
        
        self.last_spikes[trip_layer].append(triplet)
        # turn on the latest spike
        x, y = self.getxy(triplet.address, trip_layer)
        self.spike_datas[trip_layer][x][y] = 1

        # display updated image
        self.spike_imgs[trip_layer].setImage(self.spike_datas[trip_layer])


    
    def _update_plots(self):
        #import ipdb; ipdb.set_trace()
        
        # spike img
        #for img in self.spike_imgs:
        #    img.setImage(self.data[self.i])
        if len(self.history) == 0:
            return # nothing to do
        cur_time, events = self.history[self.ind]
        self.time_lab.setText(('%2.3fs'%cur_time))
        for event in events:
            evt_type = event[0]
            if evt_type == 'SPIKE': # spike occured update plots 
                triplet = Spike(time=cur_time, layer=event[1], address=event[2])
                self.update_spike_plot(triplet)
            elif evt_type == 'MEMBRANE':
                layer, layer_vals = event[1], event[2]
                self.update_membrane_plot(layer, layer_vals)
            elif evt_type == 'NEW_SPIKE_TRAIN':
                vals = event[1]
                self.update_train_plot(vals)
            elif evt_type == 'INIT_WEIGHTS':
                vals = event[1]
                self.weights_data = vals #full weights(usually comes initially)
                # visualize for each column
                for y in range(self.hidden_size):
                    self.update_weights_plot(y, self.weights_data[:, y], False)
            elif evt_type == 'UPDATE_WEIGHTS':
                column, new_vals, color_delta = event[1], event[2], event[3]
                #print self.weights_data[:, 2]
                # x - visible neuron, y - hidden neuron
                self.update_weights_plot(column,  new_vals, color_delta)
                #print np.max(self.weights_data), np.min(self.weights_data)
            
                #print 'ohoo', column


        self.ind += 1
        relative_sim = True # relative time simulation
        if relative_sim:
            delay = 10 * (cur_time - self.last_upd_time)
        else:
            delay = 10

        if self.ind < len(self.history):
            QtCore.QTimer.singleShot(delay, self._update_plots)
        self.last_upd_time = cur_time
            
    def getxy(self, address, layer):
        # convert 1d neuron coordinate to 2d
        return (address / self.layer_sizes[layer][0], 
                address % self.layer_sizes[layer][1])

#db = DashBoard(1)
#db.plot_thigns()
#db.start()
