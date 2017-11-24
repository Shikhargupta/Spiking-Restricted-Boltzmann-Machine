import numpy as np

time_step = 1 # simulation step second
simulation_time = 1000 # seconds
D = 2 # potential degradation value

P_max = 10
P_min = -10
P_rest = 0

T_refr = 10

# simple snn with 1 membrane:

n_input = 3
spike_freq = [10, 15, 35]

def simulate(time_step, simulation_time, D, P_max, P_min, P_rest, T_refr, 
        n_input, spike_freq):
    trace = [P_rest] # membrane potential trace
    output_spikes = [] # times when neuron fires spikes 

    weights = np.ones(n_input)


    for time in range(0, simulation_time, time_step):
        P_cur = trace[-1] #current potential


        # check if the membrane is in refractory period 
        if len(output_spikes) > 0 and time - output_spikes[-1] < T_refr:
            # don't do anything
            trace.append(P_cur)
            continue

        if P_cur > P_max:
            # Fire
            output_spikes.append(time)
            P_cur = P_rest
            trace.append(P_cur)
            continue

        if P_cur < P_min:
            P_cur = P_rest
            trace.append(P_cur)
            continue

        # increase the potential, incoming spikes
        for neuron in range(n_input):
            if time % spike_freq[neuron] == 0: # spike happens
                P_cur += weights[neuron]

        # now substract D for this time instant
        if P_cur > P_rest:
            P_cur -= D

        trace.append(P_cur)
    return trace, output_spikes
