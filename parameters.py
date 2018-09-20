import numpy as np
import tensorflow as tf
import os
import pdb

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Network parameters
    'pol_unit_type'         : 'Linear', # Policy network unit type: Type of single hidden layer unit
    'val_unit_type'         : 'Linear', # Value network unit type: Type of single hidden layer unit
    'pol_n_hidden'          : 100,      # Number of hidden units for the polocy network
    'val_n_hidden'          : 100,      # Number of hidden units for the value network
    'num_input'             : 36,       # Number of inputs to the policy network
    'EI'                    : False,     # Whether or not network has both E and I units according to Dale's law
    'exc_inh_prop'          : 0.8,       # Percentage of excitatory neurons, only relevant if par['EI'] is set to True,Literature 0.8, for EI off 1

    # Timings and rates
    'dt'                    : 10,       # Time step
    'learning_rate'         : 2e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1,      # Usually 1
    'discount_time_constant': 500,    # Time constant for discount reward in ms (per Song et al., 2017)
    'discount_coef'         : 0.1,    # Discount coefficient for the Schulman TD formula
    'entropy_cost'          : 0,        # Weight of entropy to encourage diversity in the output of the policy network
    'learning_rule'         : 'diff',   # According to what rule the learning happens: 'Supervised', 'TD'=RL, TD, 'diff' = RL, difference between real and predicted

    # Variance values
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.1,

    # Cost parameters
    'spike_cost'            : 0., #1e-7,
    'wiring_cost'           : 0.,   # Weight of L2 norm of policy network recurrent weights to encourage low connection densities

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 1024,
    'num_iterations'        : 10000,
    'iters_between_outputs' : 10
}

"""
Dependent parameters
"""
def update_dependencies():
    """
    Updates all parameter dependencies
    """
    '''
    # Policy network, E and I units
    par['pol_num_e'] = int(np.round(par['pol_n_hidden']*par['exc_inh_prop']))
    par['pol_num_i'] = par['pol_n_hidden'] - par['pol_num_e']

    par['pol_EI_list'] = np.ones(par['pol_n_hidden'], dtype=np.float64)
    par['pol_EI_list'][-par['pol_num_i']:] = -1.

    par['pol_drop_mask'] = np.ones((par['pol_n_hidden'],par['pol_n_hidden']), dtype=np.float64)
    ind_inh = np.where(par['pol_EI_list']==-1)[0]
    par['pol_drop_mask'][:, ind_inh] = 0.
    par['pol_drop_mask'][ind_inh, :] = 0.

    par['pol_EI_matrix'] = np.diag(par['pol_EI_list'])

    # Value network, E and I units
    par['val_num_e'] = int(np.round(par['val_n_hidden']*par['exc_inh_prop']))
    par['val_num_i'] = par['val_n_hidden'] - par['val_num_e']

    par['val_EI_list'] = np.ones(par['val_n_hidden'], dtype=np.float64)
    par['val_EI_list'][-par['val_num_i']:] = -1.

    par['val_drop_mask'] = np.ones((par['val_n_hidden'],par['val_n_hidden']), dtype=np.float64)
    ind_inh = np.where(par['val_EI_list']==-1)[0]
    par['val_drop_mask'][:, ind_inh] = 0.
    par['val_drop_mask'][ind_inh, :] = 0.

    par['val_EI_matrix'] = np.diag(par['val_EI_list'])
    '''

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float64(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']


# Added by Pantea
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
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    #w = np.random.uniform(0,0.25, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float64(w)

def spectral_radius(A):
    return np.max(abs(np.linalg.eigvals(A)))

update_dependencies()

print("--> Parameters successfully loaded.\n")
