"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
Modified in Feb 2018 by Pantea Moghimi to create a model for RL
Based on the  'Song, Yang, and Wang, 2017, eLife' paper
Class Model constructs two recurrent networks: the policy network and the value networks
"""
# TO ADD:
# Save the following values for both mine and Nick's Networks
# 1) ploss and vloss
# 2) total and predicted reward values
# 3) Predicted reward values over time
# 4) Output of the policy network over time

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
#import model_RL
import model_works
import pickle

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


    def run_model(self, task, stimulus, target):
        ###### Policy Netwrok########
        with tf.variable_scope('pol_cell', reuse=tf.AUTO_REUSE):
            pol_W_in = tf.get_variable('pol_W_in', initializer = par['pol_w_in0'], trainable=True, dtype=np.float64)
            pol_W_rnn = tf.get_variable('pol_W_rnn', initializer = par['pol_w_rnn0'], trainable=True, dtype=np.float64)
            pol_b_rnn = tf.get_variable('pol_b_rnn', initializer = par['pol_b_rnn0'], trainable=True, dtype=np.float64)
            pol_W_out = tf.get_variable('pol_W_out', initializer = par['pol_w_out0'], trainable=True, dtype=np.float64)
            pol_b_out = tf.get_variable('pol_b_out', initializer = par['pol_b_out0'], trainable=True, dtype=np.float64)

        ###### Value Netwrok########
        with tf.variable_scope('val_cell', reuse=tf.AUTO_REUSE):
            val_W_in_pol = tf.get_variable('val_W_in_pol', initializer = par['val_w_in0'][:,:-3], trainable=True) # Weights to be multiplied by activity pf policy network hidden units
            val_W_in_act = tf.get_variable('val_W_in_act', initializer = par['val_w_in0'][:,-3:], trainable=True) # Weights to be multiplied by selected actions
            val_W_rnn = tf.get_variable('val_W_rnn', initializer = par['val_w_rnn0'], trainable=True)
            val_b_rnn = tf.get_variable('val_b_rnn', initializer = par['val_b_rnn0'], trainable=True)
            val_W_out = tf.get_variable('val_W_out', initializer = par['val_w_out0'], trainable=True)
            val_b_out = tf.get_variable('val_b_out', initializer = par['val_b_out0'], trainable=True)

        task.run_trials(self.policy_hidden_init, self.value_hidden_init, stimulus, target)

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
            ind = v.name.find('pol_')
            if ind != -1:    # If the string pol is found in the name, this is a policy network variable
                pol_list.append(v)
            # List of variables that should be optimized for the value network
            ind = v.name.find('val_')
            if ind != -1:    # If the string val is found in the name, this is a value network variable
                val_list.append(v)

        '''
        Calculate reward and baseline values
        '''
        external_reward = tf.stop_gradient(task.reward*task.time_mask)    # This is the reward value given by the environment
        self.external_reward = external_reward
        baseline = tf.stop_gradient(task.val_out_history*task.time_mask)
        self.baseline = baseline

        ''' Calcaulte logpi '''
        action_array = tf.stop_gradient(task.action_array)
        pol_out = tf.log(tf.nn.softmax(tf.stack(task.pol_out_history), 1)+1e-7)   # Output of the policy network, a small amount added so log wouldn't get nan
        self.pol_out = pol_out
        logpi = tf.reduce_sum(pol_out*action_array, axis=1) #tf.log(tf.reduce_sum(logpi, axis=0))
        if par['learning_rule'] == 'TD':
            logpi = logpi[:-1]     # Discard last time point for some formulations
        self.logpi = logpi

        '''
        Calculate loss function for policy network based on learning rule
        '''
        # Calculate advantage if appicable
        if par['learning_rule'] == 'diff':
            # Calculate discounted future reward per Song et al.,2 017
            Mcausal = np.zeros((par['num_time_steps'], par['num_time_steps']))
            for i in range(par['num_time_steps']):
                # Mcausal[i,i:] = 1 # No discount version
                Mcausal[i,i:] = np.exp(-np.arange(par['num_time_steps']-i)/(par['discount_time_constant']//par['dt']))   # Add discount
            advantage = Mcausal@external_reward #- baseline
            self.advantage = advantage
        elif par['learning_rule'] == 'TD':
            # Advantage based on Nick and Greg's code
            Vt = baseline[:-1, :]   # Vt will have all baseline values but the last one
            Vtnext = baseline[1:, :]    # Vt+1 will have all baseline values but the first one
            advantage = external_reward[:-1, :] + par['discount_coef']*Vtnext - Vt
            self.advantage = advantage
        elif par['learning_rule'] == 'Supervised':
            self.advantage = external_reward  # Some random aassignemnt for the sake of consistency

        if par['learning_rule'] == 'diff' or par['learning_rule'] == 'TD':
            Jn = tf.reduce_sum(advantage*logpi, axis = 0)
        elif par['learning_rule'] == 'Supervised':
            Jn = -tf.square(tf.stack(task.pol_out_history) - target)
        else:
            disp('Unknown learning rule')
        # Average Jn values to get average of J
        self.J = tf.reduce_mean(Jn)
        #  Calculate average regularization term
        with tf.variable_scope('pol_cell', reuse=True):
            pol_W_rnn = tf.get_variable('pol_W_rnn', dtype=tf.float64)
        # Second norm of the recurrent weight loss, encourages sparse weights
        self.weight_loss_pol = par['wiring_cost'] * tf.norm(pol_W_rnn, ord=2) / par['batch_train_size']
        # Sum of firing rates squared (Second norm of their activity matrix), encourages sparse activity
        self.spike_loss_pol = par['spike_cost'] * tf.reduce_mean(tf.stack(task.pol_r_history))
        self.Omega_pol = self.weight_loss_pol + self.spike_loss_pol

        # Caclulate entropy
        log_pol_out = tf.log(pol_out + 1e-7)               # Log of output of the policy network
        # Sum over all p.log(p) values
        entropy = -1*tf.stop_gradient(task.time_mask)*tf.reduce_sum(pol_out*log_pol_out, axis=1)
        # Avrage across all trials and time points
        self.entropy = par['entropy_cost']*entropy
        # Calculate the loss function for policy network
        self.Loss_pol = -self.J + 0*self.Omega_pol - 0*self.entropy

        '''
        Calculate the loss function for baseline network
        '''
        val_out = tf.multiply(tf.stack(task.val_out_history), tf.stop_gradient(task.time_mask))
        if par['learning_rule'] == 'TD':
            # E will minimzie advantage, except instead of Vt, which is not differentiable, we use val_out which is the differentiable variable
            En = tf.square(external_reward[:-1, :] + par['discount_coef']*Vtnext - val_out[:-1, :])
        elif par['learning_rule'] == 'diff':
            En = tf.square(val_out - advantage - baseline)  # We are adding the baseline back, which was subtracted for calculating J
        elif par['learning_rule'] == 'Supervised':
            En = -tf.reduce_mean(val_out)   # Some arbitrary thing to get the code work for all learning rules
        else:
            disp('Unknown learning rule')
        # Average En values to get E
        self.E = tf.reduce_mean(En)
        # Calculate Omega for the value network (mentioned in equation 4)
        # Set it to zero for now
        self.Omega_val = 0
        # Calculate loss for the value network (Equation 4)
        self.Loss_val = self.E + self.Omega_val

        """
        Define optimizer, calculate gradients for both networks
        """
        val_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate']/10)
        self.val_grads_and_vars = val_opt.compute_gradients(self.Loss_val, var_list = val_list)
        self.val_train_op = val_opt.minimize(self.Loss_val*0, var_list = val_list)


        pol_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.pol_grads_and_vars = pol_opt.compute_gradients(self.Loss_pol, var_list = pol_list)
        self.pol_train_op = pol_opt.minimize(self.Loss_pol, var_list = pol_list)



def main(task, gpu_id):
    # If a gpu id has been provided, use it
    if gpu_id is not None:
        # Specify the gpu id to be used
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset TensorFlow before running anything
    tf.reset_default_graph()
    # Tensorflow finds the supported CPU and GPU devices you can use
    config = tf.ConfigProto()

    # Define a placeholder for the stimulus the agent sees
    stimulus = tf.placeholder(tf.float64, shape=[par['num_time_steps'], task.num_inputs, par['batch_train_size']])
    # Define a placeholder for the target or truth
    target = tf.placeholder(tf.float64, shape=[par['num_time_steps'], par['pol_n_output'], par['batch_train_size']])

    # Create a model for the given task object
    M = Model()
    #M = model_works.Model()
    # Build the tf structure that runs trials
    M.run_model(task, stimulus, target)
    M.optimize(task, target)

    par['num_receptive_fields'] = 1
    par['num_fix_tuned'] = 0
    par['num_rule_tuned'] = 0
    par['num_rules'] = 1
    par['variable_delay_max'] = 100
    par['mask_duration'] = 0
    par['n_output'] = 3
    par['input_mean'] = 0
    par['n_input'] = 36
    par['catch_trial_pct'] = 0
    par['rotation_match'] = 0
    stim = stm.Stimulus()

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        all_vloss = []
        for it in range(par['num_iterations']):
            # Create a batch of stimuli, stores in attribute stimulus for the task
            task.generate_trial()
            trial_info = task.trial_info
            #trial_info = stim.generate_trial()
            """
            Run the model
            """
            _, _, vloss, ploss, pol_out, reward, this_action_array, this_time_mask, baseline, advantage, logpi, pol_out_final, p, test = \
                sess.run([M.pol_train_op, M.val_train_op, M.Loss_val, M.Loss_pol, task.pol_out_history, M.external_reward, task.action_array, \
                task.time_mask, M.baseline, M.advantage, M.logpi, M.pol_out, task.p, task.test], \
                {stimulus: np.swapaxes(trial_info['neural_input'], 1, 0), target:np.swapaxes(trial_info['desired_output'], 1, 0)})
            pol_out = np.array(pol_out);
            all_vloss.append(vloss)
            if it%par['iters_between_outputs']==0:
                pdb.set_trace()
                plt.subplot(2, 2, 1)
                plt.plot(pol_out[:,:,0])
                plt.legend(['Fixate', 'match', 'Non-match'])
                if trial_info['desired_output'][1,-1,0]==1:
                    plt.title('match'+'__'+str(reward.sum()))
                elif trial_info['desired_output'][2,-1,0]==1:
                    plt.title('non-match'+'__'+str(reward.sum()))
                plt.subplot(2, 2, 2)
                temp = np.tile(this_time_mask[:,0].transpose(), [3, 1]).transpose()*this_action_array[:,:,0]
                plt.plot(temp)
                plt.subplot(2, 2, 3)
                plt.plot(baseline[:,0])
                plt.plot(reward[:,0])
                #plt.plot(this_time_mask[:,0])
                plt.plot(advantage[:,0])
                plt.title('baseline')
                plt.subplot(2, 2, 4)
                #plt.plot(np.squeeze(np.array(val_out))[:,0])
                #plt.title('his baseline')
                #plt.plot(trial_info['neural_input'][:,:,0].transpose())
                #plt.plot(np.swapaxes(trial_info['desired_output'], 1, 0)[:,:,0])
                #plt.title('Neural input')
                plt.plot(trial_info['neural_input'][:,:,0].transpose())
                plt.savefig(par['save_path']+'Iteration_'+str(it)+'.png')   # save the figure to file
                plt.close()
                print('%5d  |   Vloss: %6.6f   |   Reward: %4d' % (it, vloss, reward.sum()))
