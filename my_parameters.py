import numpy as np
import tensorflow as tf
import os
import pdb

print("--> Loading parameters...")

global par, analysis_par

"""
Independent parameters
"""

rnd_save_suffix = np.random.randint(10000)

par = {
    # Setup parameters
    'save_dir'              : './DMC/',


    # Network configuration
    #'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'pol_unit_type'             : 'STSP', # Policy network unit type: Type of single hidden layer unit
    'val_unit_type'             : 'STSP', # Value network unit type: Type of single hidden layer unit
    # acceptable values: 'Linear' (traditional)
    #                    'GRU' (Gated Recurent Units) used in Song 2017, described in (Chung et al., 2014)
    #                    'STSP' (Short Time Synaptic Plasticity) used by Nick, described in (Wang et al., 2006, Nature Neurosci.)

    # Policy network shape
    'num_input'             : 36,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    #'n_hidden'              : 100,  # Number of hidden units in the recurrent network
    #'n_output'              : 3,    # Number of outputs for the policy networks
    # Value network shape
    # Reminder: Value network only has one output: expected reward (Song et al., 2017)
    'num_motion_tuned'      : 36,
    #'num_fix_tuned'         : 0,
    #'num_rule_tuned'        : 0,
    #'n_hidden'              : 100,  # Number of hidden units in the recurrent network
    #'n_output'              : 3,

    # Timings and rates
    'dt'                    : 10,   # Time step in ms (per Song et al., 2017)
    'learning_rate'         : 4e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 0.1,         # Usually 1
    'discount_time_constant': 10000,    # Time constant for discount reward in ms (per Song et al., 2017)

    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.05,
    'noise_rnn_sd'          : 0.1,

    # Tuning function data
    # 'num_motion_dirs'       : 8,
    #'tuning_height'         : 4,        # magnitutde scaling factor for von Mises
    #'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-7,      # Weight of L2 norm of spiking activity in the cost function to encourage sparsity
    'weight_cost'           : 0,      # Weight of L2 norm of policy network recurrent weights to encourage low connection densities (added by Pantea)

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 256,
    'num_iterations'        : 10000,
    'iters_between_outputs' : 100,

    'num_receptive_fields'  : 1,
    'num_motion_dirs'       : 8,
    #'num_motion_tuned'      : 36,
    #'num_fix_tuned'         : 1,
    'num_rule_tuned'        : 0,
    'num_rules'             : 1,
    'tuning_height'         : 4,
    'kappa'                 : 2,
    'trial_type'            : 'DMC',
    'variable_delay_max'    : 100,
    'var_delay'             : False,
    'decoding_test_mode'    : True,
    'catch_trial_pct'       : 0,
    'rotation_match'        : 0
}


"""
Dependent parameters
"""

def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of input neurons
    #par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    #par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    # Possible rules based on rule type values
    #par['possible_rules'] = [par['num_receptive_fields'], par['num_categorizations']]
    '''
    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])
    '''
    # Membrane time constant of RNN neurons, alpha=dt/tau, tau being time constant of the network
    # This is the alpha in the PLOS paper
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step, equation (7) of the PLOS paper
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    # The standard deviation of the Gaussian noise added to input
    # Equation (11) of the PLOS paper
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################
    # Initial hidden unit activity for the policy network
    par['policy_h_init'] = 0.1*np.ones((par['pol_n_hidden'], par['batch_train_size']), dtype=np.float64)
    # Initial hidden unit activity for the value network
    par['value_h_init'] = 0.1*np.ones((par['val_n_hidden'], par['batch_train_size']), dtype=np.float64)

    # Initialize synaptic parameters if applicable
    # Policy network
    if par['pol_unit_type'] == 'STSP':
        pol_syn_par = set_synaptic_parameters(par['pol_n_hidden'])
        par['pol_syn_par'] = pol_syn_par
    # Value network
    if par['val_unit_type'] == 'STSP':
        val_syn_par = set_synaptic_parameters(par['val_n_hidden'])
        par['val_syn_par'] = val_syn_par

def set_synaptic_parameters(n_hidden):

    """
    Setting up synaptic parameters for a given number of hidden units (n_hidden)
    1 = facilitating
    2 = depressing
    Output: A dictionary object containing all the necessary arrays
    """
    syn_par = {}    # Dictionary object containing all pertaining parameters
    # even numbers facilitating (i.e. 1), odd numbers depressing (i.e. 2)
    syn_par['synapse_type'] = np.ones(n_hidden, dtype=np.int8)
    syn_par['ind'] = range(1,n_hidden,2)
    syn_par['synapse_type'][syn_par['ind']] = 2

    syn_par['alpha_stf'] = np.ones((n_hidden, 1), dtype=np.float32)
    syn_par['alpha_std'] = np.ones((n_hidden, 1), dtype=np.float32)
    syn_par['U'] = np.ones((n_hidden, 1), dtype=np.float32)

    # initial synaptic values
    syn_par['syn_x_init'] = np.zeros((n_hidden, par['batch_train_size']), dtype=np.float32)
    syn_par['syn_u_init'] = np.zeros((n_hidden, par['batch_train_size']), dtype=np.float32)

    for i in range(n_hidden):
        if syn_par['synapse_type'][i] == 1:
            syn_par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            syn_par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            syn_par['U'][i,0] = 0.15
            syn_par['syn_x_init'][i,:] = 1
            syn_par['syn_u_init'][i,:] = syn_par['U'][i,0]

        elif syn_par['synapse_type'][i] == 2:
            syn_par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            syn_par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            syn_par['U'][i,0] = 0.45
            syn_par['syn_x_init'][i,:] = 1
            syn_par['syn_u_init'][i,:] = syn_par['U'][i,0 ]
    return syn_par

def initialize(dims, connection_prob):
    #w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w = np.random.uniform(-0.25,0.25, size=dims)
    #w *= (np.random.rand(*dims) < connection_prob)
    return np.float64(w)

def spectral_radius(A):
    return np.max(abs(np.linalg.eigvals(A)))

#update_dependencies()

print("--> Parameters successfully loaded.\n")
