import numpy as np
from collections import defaultdict
from uvnn.classifiers.misc import random_weight_matrix

sim_time = 600
frag_time = 100
def gen_spike(periods = [10, 40], sim_time = 200, frag_time=100):
    # build a spike train with alternating periods
    ret = []
    t_now = 0
    cur_period = 0
    while (t_now < sim_time):
        fragment = range(t_now, t_now + frag_time, periods[cur_period])
        ret.extend(fragment)
        cur_period = (cur_period + 1) % len(periods)        
        t_now = fragment[-1] + periods[cur_period]
    return ret

train1 = gen_spike([5, 21], sim_time, frag_time)
train2 = gen_spike([21, 5], sim_time, frag_time)
train3 = gen_spike([5, 21], sim_time, frag_time)
train4 = gen_spike([21, 5], sim_time, frag_time)
spike_trains = [train1, train2]#, train3, train4]

conf = {'time_step': 1, # simulation step second
       'simulation_time':sim_time, # seconds
        'D':0.01, # potential degradation value
        'p_max':3,
        'p_min':-2,
        'p_rest':0,
        't_refr':14,
        'n_input':4,
        'n_output':2,
        'spike_trains':spike_trains,
        'w_max':15,
        'w_min':-15,
        'sigma':0.1,
        'margin':0,
        'windowsize':10,
        'tau_plus':20,
        'tau_minus':20,
        'a_plus':0.2,
        'a_minus':-0.2,
        'd_neighbours':0.1
       }


def get_presynaptics(time, spike_dict):
    ''' Get which neurons are firing on provided time

    Right now we feed spike_freq but it can be array of spikes trains later

    Args: 
        time: current clock cycle
        spike_dict: <time, [n1, n2..]>
    Return:
        array of currently active active neurons
    '''
    #TODO(remove this function if it stays too short)
    return spike_dict[time]

def get_spike_dict(spike_trains):
    ''' build up a map with key - time, value - array of neurons which spike
    at this moment 
    
    Args: 
        spike_trains: arrays of spike trains for each neuron

    Return:
        Dictionary of type <time, [n1, n2,...] > where n1, n2 .. are 
        neuron indicies which fire on timestep time:
    '''
    spike_dict = defaultdict(list)
    for neuron, times in enumerate(spike_trains):
        for spike_time in times:
            spike_dict[spike_time].append(neuron)
    return spike_dict


def simulate(time_step, simulation_time, D, p_max, p_min, p_rest, t_refr, 
        n_input, spike_trains, n_output, tau_plus, tau_minus, a_plus, a_minus,
        w_max, w_min, windowsize, margin, sigma, d_neighbours):
        
    init_potentials = np.zeros(n_output)
    init_potentials.fill(p_rest)

    traces = [init_potentials] # membranes potentials
    
    # all output spikes for each neuron
    out_spikes = [set() for _ in range(n_output)]



    # we keep the times when each neuron fires spikes 
    out_spikes_last = np.zeros(n_output) 
    out_spikes_last.fill(-10000)
     
    weights = 2 * np.random.rand(n_input, n_output) # initialize weights
    #weights = random_weight_matrix(n_input, n_input) 
    # build up a dict which will help us getting spikes on timestep
    spike_dict = get_spike_dict(spike_trains)
    w_trace = []

    for time in range(0, simulation_time, time_step): 
        w_trace.append(weights.copy()) 
        cur_potentials = traces[-1]
        new_potentials = np.copy(cur_potentials)

        # fire neurons which are above threshold
        fired = cur_potentials > p_max

        # depress potentials of neighbours
        if np.any(fired):
            #nz = np.where(fired > 0)
            new_potentials -= d_neighbours
            new_potentials[fired] += d_neighbours


        for post_neuron in np.nonzero(fired)[0]:
            out_spikes[post_neuron].add(time)
            out_spikes_last[post_neuron] = time
        new_potentials[fired] = p_rest

        # update weights for presynaptic neurons (positively correlated)
        already_updated = set() # keep already updated neurons not to update twice
        if np.any(fired):
            # TODO consider refactor 3 time nesting
            #import ipdb; ipdb.set_trace()
            for pre_time in range(time - windowsize, time - margin):
                delta_time = time - pre_time
                pre_neurons = get_presynaptics(pre_time, spike_dict)
                # distract neurons already updated
                pre_neurons = list(set(pre_neurons) - already_updated)
                if len(pre_neurons) > 0: 
                    dw = a_plus * np.exp(delta_time / float(tau_plus))
                    d_increase = sigma * dw * (w_max - weights[pre_neurons, fired])
                    print 'Increasing ... ', dw, d_increase
                    weights[pre_neurons, fired] += d_increase
                    already_updated = already_updated | set(pre_neurons) # unite

        
        # depolarize strongly negative neurons
        strongly_negative = cur_potentials < p_min
        new_potentials[strongly_negative] = p_rest

            
        #get postsynaptic neurons which  are not in refractory period
        # and add weights from presynaptic
        post_neurons = out_spikes_last < time - t_refr
        
        # get presynaptic neurons which fires now
        pre_neurons = get_presynaptics(time, spike_dict)

        for post_neuron in np.nonzero(post_neurons)[0]:
            for pre_neuron in pre_neurons:
                new_potentials[post_neuron] += weights[pre_neuron][post_neuron]
        
        
        # now update wieghts for postsyanptic neurons (negatively correlated)
        if np.any(pre_neurons):
            already_updated = set()
            for pre_time in range(time - windowsize, time - margin):
                for post_neuron in range(n_output):
                    # check if already updated 
                    if post_neuron in already_updated:
                        continue
                    # check if post_neuron was fired before pre_neuron
                    if pre_time in out_spikes[post_neuron]:
                        delta_time = time - pre_time
                        dw = a_minus * np.exp(delta_time / float(tau_minus))
                        d_decrease = sigma * dw * (weights[pre_neurons, post_neuron]-w_min)
                        print 'Decreasing ...', dw, d_decrease
                        weights[pre_neurons, post_neuron] += d_decrease
                    already_updated.add(post_neuron)
        


        
        # degrade potentials > p_rest
        to_degrade = new_potentials > p_rest
        new_potentials[to_degrade] -= D


        traces.append(new_potentials)  # keep potential trace

    return traces, out_spikes, w_trace

simulate(**conf)
