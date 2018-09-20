# Task tools
# This module contains a set of functions used for execution of any tasks
# List of functions implemented:
#   1) pol_cell: the policy (recurrent) neural network main computation
#   2) val_cell: the value (recurrent) neural network main computation
#   3) action_choice: Chooses an action given the probabilities of each action


import numpy as np
import tensorflow as tf
import math
from parameters import *
import pdb


def pol_cell(pol_u, pol_x, **kwargs):
    # Calculate firing rate (r) at time t-1 from current (x) at time t-1
    pol_r = tf.nn.relu(pol_x)

    # Get the recurrnet and input weights
    with tf.variable_scope('pol_cell', reuse=True):
            pol_W_in = tf.get_variable('pol_W_in', dtype=tf.float64)
            pol_W_rnn = tf.get_variable('pol_W_rnn', dtype=tf.float64)
            pol_b_rnn = tf.get_variable('pol_b_rnn', dtype=tf.float64)
            pol_W_out = tf.get_variable('pol_W_out', dtype=tf.float64)
            pol_b_out = tf.get_variable('pol_b_out', dtype=tf.float64)

    ####### Linear networks #######
    if par['pol_unit_type'] == 'Linear':
        # Implementation of a Linear traditional recurrent unit (without the GRU)
        # Calculate hidden state activity (x and r) at the next time step
        # Equations 7, 8 and 9 form Song et al., 2016 (traditional RNNs)
        # Recurrent noise
        pol_rec_noise = tf.random_normal([par['pol_n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float64)
        # Caclulate next x
        pol_next_x = (1-par['alpha_neuron'])*pol_r + \
                    par['alpha_neuron']*(pol_W_rnn@ pol_r + pol_W_in@pol_u + pol_b_rnn) + pol_rec_noise
        # Firing rates (r)
        pol_next_r = tf.nn.relu(pol_next_x)
        # Calculate output
        pol_out = pol_W_out@pol_next_r + pol_b_out
        pol_other_params = {}   # No other params for a linear unit
    return pol_next_x, pol_next_r, pol_out, pol_other_params

def val_cell(val_u_hpol, val_u_act, val_x, **kwargs):
    # Calculate firing rate (r) at time t-1 from current (x) at time t-1
    val_r = tf.nn.relu(val_x)
    # Get the recurrnet and input weights
    with tf.variable_scope('val_cell', reuse=True):
            #val_W_in = tf.get_variable('val_W_in', dtype=tf.float64)
            val_W_in_pol = tf.get_variable('val_W_in_pol', dtype=tf.float64) # Weights to be multiplied by activity pf policy network hidden units
            val_W_in_act = tf.get_variable('val_W_in_act', dtype=tf.float64) # Weights to be multiplied by selected actions
            val_W_rnn = tf.get_variable('val_W_rnn', dtype=tf.float64)
            val_b_rnn = tf.get_variable('val_b_rnn', dtype=tf.float64)
            val_W_out = tf.get_variable('val_W_out', dtype=tf.float64)
            val_b_out = tf.get_variable('val_b_out', dtype=tf.float64)

    ####### Linear networks #######
    if par['val_unit_type'] == 'Linear':
        # Implementation of a Linear traditional recurrent unit (without the GRU)
        # Calculate hidden state activity (x and r) at the next time step
        # Equations 7, 8 and 9 form Song et al., 2016 (traditional RNNs)
        # Recurrent noise
        val_rec_noise = tf.random_normal([par['val_n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float64)
        # Caclulate next x
        val_next_x = (1-par['alpha_neuron'])*val_r + \
                    par['alpha_neuron']*(tf.matmul(val_W_rnn,val_r) + \
                    tf.matmul(val_W_in_pol,val_u_hpol) + tf.matmul(val_W_in_act,val_u_act) + \
                    val_b_rnn) + val_rec_noise
        # Firing rates (r)
        val_next_r = tf.nn.relu(val_next_x)
        # Calculate output
        val_out = val_W_out @ val_next_r + val_b_out
        val_other_params = {}   # No other params for a linear unit
    return val_next_x, val_next_r, val_out, val_other_params
