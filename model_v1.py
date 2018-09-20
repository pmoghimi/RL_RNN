"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
Modified in Feb 2018 by Pantea Moghimi to create a model for RL
Based on the  'Song, Yang, and Wang, 2017, eLife' paper
Class Model constructs two recurrent networks: the policy network and the value networks
"""

import tensorflow as tf
import numpy as np
import stimulus as stm
import time
import analysis
from parameters import *
import pdb
# Module containing all the tasks
from Tasks import *
import matplotlib.pyplot as plt
import model_RL

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#print('Using EI Network:\t', par['EI'])
#print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""

class Model:
    # The parameter requried to isntantiate a Model object is the task the Networks
    # are to be trained on

    def __init__(self):
        # Initialize hidden unit activity of poicy and value networks
        # 1) Policy network
        #self.policy_hidden_init = tf.constant(par['policy_h_init'], dtype=np.float64)
        self.policy_hidden_init = tf.constant(par['policy_h_init'], dtype=np.float64)
        # 2) Value network
        #self.value_hidden_init = tf.constant(par['value_h_init'], dtype=np.float64)
        self.value_hidden_init = tf.constant(par['value_h_init'], dtype=np.float64)
        # Initialize utilization and facilitation values if network type is STSP
        # Policy network
        if par['pol_unit_type'] == 'STSP':
            self.pol_syn_x_init = tf.constant(par['pol_syn_par']['syn_x_init'], dtype=np.float64)
            self.pol_syn_u_init = tf.constant(par['pol_syn_par']['syn_u_init'], dtype=np.float64)
        # Value network
        if par['val_unit_type'] == 'STSP':
            self.val_syn_x_init = tf.constant(par['val_syn_par']['syn_x_init'], dtype=np.float64)
            self.val_syn_u_init = tf.constant(par['val_syn_par']['syn_u_init'], dtype=np.float64)


    def run_model(self, task, stimulus, truth):
        """
        Run the policy and value reccurent networks through a batch of trials
        History of hidden state activity stored in:
            self.policy_hidden_state_hist for the policy network
            self.value_hiden_state_hist for the value network
            stimulus: The batch of trials that constitute the stimulus
        """

        """
        Initialize weights and biases for both networks
        """
        ###### Policy Netwrok########
        # Policy network input and recurrent weights
        with tf.variable_scope('policy_rnn_cell', reuse=tf.AUTO_REUSE):
            pol_W_in = tf.get_variable('pol_W_in', initializer = par['pol_w_in0'], trainable=True, dtype=np.float64)
            pol_W_rnn = tf.get_variable('pol_W_rnn', initializer = par['pol_w_rnn0'], trainable=True, dtype=np.float64)
            pol_b_rnn = tf.get_variable('pol_b_rnn', initializer = par['pol_b_rnn0'], trainable=True, dtype=np.float64)
        # Policy network output weights
        with tf.variable_scope('policy_output', reuse=tf.AUTO_REUSE):
            pol_W_out = tf.get_variable('pol_W_out', initializer = par['pol_w_out0'], trainable=True, dtype=np.float64)
            pol_b_out = tf.get_variable('pol_b_out', initializer = par['pol_b_out0'], trainable=True, dtype=np.float64)
        # Policy network update and reset gate weghits (is applicable)
        if par['pol_unit_type']=='GRU':
            # Weights for calculating update gate values
            with tf.variable_scope('policy_update', reuse=tf.AUTO_REUSE):
                pol_W_in_update = tf.get_variable('pol_W_in_update', initializer = par['pol_w_in0'], trainable=True, dtype=np.float64)
                pol_W_rnn_update = tf.get_variable('pol_W_rnn_update', initializer = par['pol_w_rnn0'], trainable=True, dtype=np.float64)
                pol_b_rnn_update = tf.get_variable('pol_b_rnn_update', initializer = par['pol_b_rnn0'], trainable=True, dtype=np.float64)
            # Weights for calculating reset gate values
            with tf.variable_scope('policy_reset', reuse=tf.AUTO_REUSE):
                pol_W_in_reset = tf.get_variable('pol_W_in_reset', initializer = par['pol_w_in0'], trainable=True, dtype=np.float64)
                pol_W_rnn_reset = tf.get_variable('pol_W_rnn_reset', initializer = par['pol_w_rnn0'], trainable=True, dtype=np.float64)
                pol_b_rnn_reset = tf.get_variable('pol_b_rnn_reset', initializer = par['pol_b_rnn0'], trainable=True, dtype=np.float64)


        ###### Value Netwrok########
        # Value network input and recurrent weights
        with tf.variable_scope('value_rnn_cell', reuse=tf.AUTO_REUSE):
            v_W_in = tf.get_variable('val_W_in', initializer = par['val_w_in0'], trainable=True)
            v_W_rnn = tf.get_variable('val_W_rnn', initializer = par['val_w_rnn0'], trainable=True)
            v_b_rnn = tf.get_variable('val_b_rnn', initializer = par['val_b_rnn0'], trainable=True)
        # Value network output weights
        with tf.variable_scope('value_output', reuse=tf.AUTO_REUSE):
            val_W_out = tf.get_variable('val_W_out', initializer = par['val_w_out0'], trainable=True)
            val_b_out = tf.get_variable('val_b_out', initializer = par['val_b_out0'], trainable=True)
        # Value network update and reset gate weghits (is applicable)
        if par['val_unit_type']=='GRU':
            # Weights for calculating update gate values
            with tf.variable_scope('value_update', reuse=tf.AUTO_REUSE):
                val_W_in_update = tf.get_variable('val_W_in_update', initializer = par['val_w_in0'], trainable=True)
                val_W_rnn_update = tf.get_variable('val_W_rnn_update', initializer = par['val_w_rnn0'], trainable=True)
                val_b_rnn_update = tf.get_variable('val_b_rnn_update', initializer = par['val_b_rnn0'], trainable=True)
            # Weights for calculating reset gate values
            with tf.variable_scope('value_reset', reuse=tf.AUTO_REUSE):
                val_W_in_reset = tf.get_variable('val_W_in_reset', initializer = par['val_w_in0'], trainable=True)
                val_W_rnn_reset = tf.get_variable('val_W_rnn_reset', initializer = par['val_w_rnn0'], trainable=True)
                val_b_rnn_reset = tf.get_variable('val_b_rnn_reset', initializer = par['val_b_rnn0'], trainable=True)


        '''
        Run the trials over time and get chosen actions, rewards and activity and output of both network
        The inputs to the run_trials function are initial activity of hidden units
        for both policy and value networks
        '''
        extra_args = {}
        # Policy network
        if par['pol_unit_type'] == 'STSP':
            extra_args['pol_syn_x_init'] = self.pol_syn_x_init
            extra_args['pol_syn_u_init'] = self.pol_syn_u_init
        # Value network
        if par['val_unit_type'] == 'STSP':
            extra_args['val_syn_x_init'] = self.val_syn_x_init
            extra_args['val_syn_u_init'] = self.val_syn_u_init
        task.run_trials(self.policy_hidden_init, self.value_hidden_init, stimulus, truth, extra_args)

    def optimize(self, task, target):

        """
        Calculate the loss functions and optimize the weights
        """
        # Get a list of all trainable variables
        variables_names = [v for v in tf.trainable_variables()]
        pol_list = []
        val_list = []
        for v in variables_names:
            # List of variables that should be optimized for the policy network
            ind = v.name.find('pol')
            if ind != -1:    # If the string pol is found in the name, this is a policy network variable
                pol_list.append(v)
            # List of variables that should be optimized for the value network
            ind = v.name.find('val')
            if ind != -1:    # If the string val is found in the name, this is a value network variable
                val_list.append(v)

        '''
        Calculate the loss function dependent on the policy netwokrk parameters
        Equation (2) from Song et al., 2017
        '''
        # Calculate J (equation 22 bur also baseline will be subtracted):
        # 1) Discard reward at time points that are to be excluded
        reward = tf.multiply(task.reward, task.time_mask)
        #reward = task.reward
        #reward = tf.cast(reward, dtype=tf.float64)  # Make reward a float tensor so it can be multiplied by other float factors
        # 2) Apply discount (Page 17, Song et al., 2017)
        # Transform temporal discount into a format that can be multiplied by reward
        discount = np.transpose(np.tile(np.array(task.discount), (par['batch_train_size'], 1)))
        #discount = 1
        # Multiply by discount
        reward = tf.multiply(reward, discount)
        self.reward = reward
        # 3) Multiply reward by logpi to get the first term in J (i.e. reward portion)
        #pdb.set_trace()
        #task.logpi1 = tf.cumprod(task.logpi, axis=0)
        task.cumsum_logpi = tf.cumsum(task.logpi, axis=0)
        self.J1 = tf.multiply(reward, task.cumsum_logpi)
        #self.J1 = reward
        # 4) Discard output of the value network (predicted reward) at time points that are to be excluded
        baseline = tf.multiply(tf.stack(task.val_out_history), task.time_mask)
        self.baseline = baseline
        # 5) Multiply output of the value network (predicted reward) by logpi to get teh second term in J (i.e. baseline subtraction portion)
        self.J2 = tf.multiply(baseline, task.cumsum_logpi) # I think Song et al. used logpi here and not the cumsum!
        # 6) Subtract J2 from J1 and calculate total reward (sum across time) for each trial to calculate all Jn values
        self.Jn = tf.reduce_sum(self.J1 - self.J2, axis=0)
        # Average Jn values to get average of J
        self.J = tf.reduce_mean(self.Jn)
        '''
        # Calculate J as supervised
        y_hat = tf.stack(task.pol_out_history0)
        self.y_hat = y_hat; self.target = target;
        #self.J = tf.reduce_mean(tf.square(y_hat - target), axis=0)
        #pdb.set_trace()
        self.J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = target, dim = 1))
        self.J = -1*tf.reduce_mean(self.J)
        '''
        # 7) Calculate average regularization term (mentioned as Omega in equation 2)
        with tf.variable_scope('policy_rnn_cell', reuse=True):
            pol_W_rnn = tf.get_variable('pol_W_rnn', dtype=tf.float64)
        # Second norm of the recurrent weight loss, encourages sparse weights
        self.weight_loss_pol = par['weight_cost'] * tf.norm(pol_W_rnn, ord=2) / par['batch_train_size']
        # Sum of firing rates squared (Second norm of their activity matrix), encourages sparse activity
        self.spike_loss_pol = par['spike_cost'] * tf.norm(tf.stack(task.pol_r_history), ord=2) / par['pol_n_hidden']
        self.Omega_pol = 0 #self.weight_loss_pol + self.spike_loss_pol

        # Caclulate netropy
        #pdb.set_trace()
        pol_out = tf.stack(task.pol_out_history) + 1e-6   # Output of the policy network, a small amount added so log wouldn't get nan
        log_pol_out = tf.log(pol_out)               # Log of output of the policy network
        # Multiply output and its log
        entropy = tf.multiply(pol_out, log_pol_out)
        # Sum over all the outputs
        entropy = tf.reduce_sum(entropy, axis=1)
        # Apply time mask
        entropy = tf.multiply(entropy, task.time_mask)
        # Sum across time
        entropy = tf.reduce_sum(entropy, axis=0)
        # Average across trials
        entropy = -tf.reduce_mean(entropy)
        self.entropy = entropy
        # 8) Calculate the loss function for policy network (Equation 2)
        self.Loss_pol = -self.J + self.Omega_pol - 0.001*self.entropy

        '''
        Calculate the loss function dependent on the value netwokrk parameters
        Equation (4) from Song et al., 2017
        '''

        # 1) Calculate En (Equation 5)
        # Sum of squared of differences averaged across all time points
        self.En = tf.reduce_mean(tf.square(baseline - reward), axis=0)
        # Average En values to get E
        self.E = tf.reduce_mean(self.En)
        # 2) Calculate Omega for the value network (mentioned in equation 4)
        # Set it to zero for now
        self.Omega_val = 0
        # 3) Calculate loss for the value network (Equation 4)
        self.Loss_val = self.E + self.Omega_val


        """
        Define optimizer, calculate and gradient the the value network
        """

        # Optimizer for value network
        val_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        # Gradient of the value network
        self.val_grads_and_vars = val_opt.compute_gradients(self.Loss_val, var_list = val_list)

        # Gradient normalization (clipping)
        self.val_capped_gvs = []
        for grad, var in self.val_grads_and_vars:
            if not str(type(grad)) == "<class 'NoneType'>":
                self.val_capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        # Apply normalized gradients
        self.val_train_op = val_opt.apply_gradients(self.val_capped_gvs)

        """
        Define optimizer, calculate and gradient the the policy network
        """
        # Optimizer for policy network
        pol_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        # Gradient of the policy network
        #self.pol_grads_and_vars = pol_opt.compute_gradients(self.Loss_pol, var_list = pol_list)
        self.pol_grads_and_vars = pol_opt.compute_gradients(self.Loss_pol, var_list = pol_list)

        # Gradient normalization (clipping)
        self.pol_capped_gvs = []
        for grad, var in self.pol_grads_and_vars:
            if not str(type(grad)) == "<class 'NoneType'>":
                self.pol_capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        # Apply normalized gradients
        #pdb.set_trace()
        self.pol_train_op = pol_opt.apply_gradients(self.pol_capped_gvs)


def main(task):
    """
    This function will take the task object as input and
    creates a model object to learn the task
    It would run the iterations
    At each iteration, a new batch of trials will be created
    """
    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    # Tensorflow finds the supported CPU and GPU devices you can use
    config = tf.ConfigProto()

    trial_length = par['num_time_steps']


    # Calculate shape of the stimulus for this task
    # Define a placeholder for the stimulus the agent sees
    #stimulus = tf.placeholder(tf.float64, shape=[task.total_dur, task.num_inputs, par['batch_train_size']])
    stimulus = tf.placeholder(tf.float64, shape=[trial_length, task.num_inputs, par['batch_train_size']])
    # Define a placeholder for the truth or correct answer about each trial
    truth = tf.placeholder(tf.float64, shape=par['batch_train_size'])

    # A TEMPORARY placeholder for target
    #target = tf.placeholder(tf.float64, shape=[task.total_dur, 3, par['batch_train_size']])
    target = tf.placeholder(tf.float64, shape=[trial_length, 3, par['batch_train_size']])
    # Create a model for the given task object
    M = Model()
    # Build the tf structure that runs trials
    M.run_model(task, stimulus, truth)
    M.optimize(task, target)

    # Create a model from Nick's code
    stim = stm.Stimulus()
    n_input=task.num_inputs
    '''
    mask = tf.placeholder(tf.float64, shape=[task.total_dur, par['batch_train_size']])
    x = tf.placeholder(tf.float64, shape=[n_input, task.total_dur, par['batch_train_size']])  # input data
    target2 = tf.placeholder(tf.float64, shape=[3, task.total_dur, par['batch_train_size']])  # input data
    actual_reward = tf.placeholder(tf.float64, shape=[task.total_dur,par['batch_train_size']])
    pred_reward = tf.placeholder(tf.float64, shape=[task.total_dur, par['batch_train_size']])
    actual_action = tf.placeholder(tf.float64, shape=[task.total_dur, 3, par['batch_train_size']])
    '''
    mask = tf.placeholder(tf.float64, shape=[trial_length, par['batch_train_size']])
    x = tf.placeholder(tf.float64, shape=[n_input, trial_length, par['batch_train_size']])  # input data
    target2 = tf.placeholder(tf.float64, shape=[3, trial_length, par['batch_train_size']])  # input data
    actual_reward = tf.placeholder(tf.float64, shape=[trial_length,par['batch_train_size']])
    pred_reward = tf.placeholder(tf.float64, shape=[trial_length, par['batch_train_size']])
    actual_action = tf.placeholder(tf.float64, shape=[trial_length, 3, par['batch_train_size']])
    M_Nick = model_RL.Model(x, target2, actual_reward, pred_reward, actual_action, mask)
    #M_Nick.run_model(task, stimulus, truth)
    #M_Nick.optimize(task, target)

    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        vloss = np.zeros((1, par['num_iterations']))
        ploss = np.zeros((1, par['num_iterations']))
        perf = np.zeros((1, par['num_iterations']))
        for it in range(par['num_iterations']):
            # Create a batch of stimuli, stores in attribute stimulus for the task
            #task.create_stimulus()
            # generate batch of batch_train_size with Nick's code
            trial_info = stim.generate_trial()
            """
            Run the model
            """

            my_truth = np.zeros(par['batch_train_size'])
            my_truth[trial_info['desired_output'][1,-1,:]==1] = 1   # Match trials
            my_truth[trial_info['desired_output'][2,-1,:]==1] = 2   # Non_match trials
            _, _, vloss[0, it], ploss[0, it], pol_grads, pol_out, pol_out0, actions, logpi, my_reward, action_array, time_mask, cumsum_logpi, pol_r, temp1, temp2, ideal, my_baseline, entropy = \
                sess.run([M.pol_train_op, M.val_train_op, M.Loss_val, M.Loss_pol, M.pol_capped_gvs,task.pol_out_history, task.pol_out_history0, task.actions, task.logpi, M.reward, task.action_array, task.time_mask, task.cumsum_logpi, \
                task.pol_r_history, task.temp1, task.temp2, task.ideal, M.baseline, \
                M.entropy], {stimulus: np.swapaxes(trial_info['neural_input'], 1, 0), truth: my_truth, target: np.swapaxes(trial_info['desired_output'], 1, 0)})
            # Run Nick's model
            pol_out, val_out, pol_rnn, action, stacked_mask, reward = sess.run([M_Nick.pol_out, M_Nick.val_out, M_Nick.h_pol, M_Nick.action, \
                 M_Nick.stacked_mask,M_Nick.reward], {x: trial_info['neural_input'], target2: trial_info['desired_output'], mask: trial_info['train_mask']})

            trial_reward = np.squeeze(np.stack(reward))
            trial_action = np.stack(action)

            _, _, pol_loss, val_loss = sess.run([M_Nick.train_pol, M_Nick.train_val, M_Nick.pol_loss, M_Nick.val_loss], \
                {x: trial_info['neural_input'], target2: trial_info['desired_output'], mask: trial_info['train_mask'], \
                actual_reward: trial_reward, pred_reward: np.squeeze(val_out), actual_action:trial_action })

            pol_out = np.array(pol_out)
            pol_out0 = np.array(pol_out0)
            temp1 = np.array(temp1); temp2 = np.array(temp2)

            if it%100==0:
                fig = plt.plot(pol_out[:,:,0])
                plt.legend(['Fixate', 'match', 'Non-match'])
                plt.title(str(my_truth[0]))
                plt.savefig('Iteration_'+str(it)+'.png')   # save the figure to file
                plt.close()
                print('%6d,     %6.1f,   %6.1f,     %6.1f,  %6.1f,  %6.2f' % (it, my_reward.sum(), my_baseline.sum(), ploss[0, it], vloss[0, it], entropy))
                print('%6d,     %6.1f,   %6.1f,     %6.1f,  %6.1f' % (it, np.array(trial_reward).sum(), np.array(val_out).sum(), pol_loss, val_loss))
                #pdb.set_trace()
            #plt.plot(pol_out[:,:,0]); plt.show()
            #if np.isnan(ploss[0, it]):
            #    pdb.set_trace()
            #if it>=1000:
            #    pdb.set_trace()

        pdb.set_trace()
        a = 5

def eval_weights():
    # Policy network input and recurrent weights
    with tf.variable_scope('policy_rnn_cell', reuse=True):
        pol_W_in = tf.get_variable('pol_W_in', dtype=tf.float64)
        pol_W_rnn = tf.get_variable('pol_W_rnn', dtype=tf.float64)
        pol_b_rnn = tf.get_variable('pol_b_rnn', dtype=tf.float64)
    # Policy network output weights
    with tf.variable_scope('policy_output', reuse=True):
        pol_W_out = tf.get_variable('pol_W_out', dtype=tf.float64)
        pol_b_out = tf.get_variable('pol_b_out', dtype=tf.float64)

    weights = {
        'pol_w_in'  : pol_W_in.eval(),
        'pol_w_rnn' : pol_W_rnn.eval(),
        'pol_w_out' : pol_W_out.eval(),
        'pol_b_rnn' : pol_b_rnn.eval(),
        'pol_b_out'  : pol_b_out.eval()
    }

    return weights
