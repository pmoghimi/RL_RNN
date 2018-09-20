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
import model_RL
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


    def run_model(self, task, stimulus, truth, target):
        ###### Policy Netwrok########
        # Policy network input and recurrent weights
        with tf.variable_scope('pol_rnn_cell', reuse=tf.AUTO_REUSE):
            pol_W_in = tf.get_variable('pol_W_in', initializer = par['pol_w_in0'], trainable=True, dtype=np.float64)
            pol_W_rnn = tf.get_variable('pol_W_rnn', initializer = par['pol_w_rnn0'], trainable=True, dtype=np.float64)
            pol_b_rnn = tf.get_variable('pol_b_rnn', initializer = par['pol_b_rnn0'], trainable=True, dtype=np.float64)
        # Policy network output weights
        with tf.variable_scope('pol_output', reuse=tf.AUTO_REUSE):
            pol_W_out = tf.get_variable('pol_W_out', initializer = par['pol_w_out0'], trainable=True, dtype=np.float64)
            pol_b_out = tf.get_variable('pol_b_out', initializer = par['pol_b_out0'], trainable=True, dtype=np.float64)

        ###### Value Netwrok########
        # Value network input and recurrent weights
        with tf.variable_scope('val_rnn_cell', reuse=tf.AUTO_REUSE):
            val_W_in_pol = tf.get_variable('val_W_in_pol', initializer = par['val_w_in0'][:,:-3], trainable=True) # Weights to be multiplied by activity pf policy network hidden units
            val_W_in_act = tf.get_variable('val_W_in_act', initializer = par['val_w_in0'][:,-3:], trainable=True) # Weights to be multiplied by selected actions
            val_W_rnn = tf.get_variable('val_W_rnn', initializer = par['val_w_rnn0'], trainable=True)
            val_b_rnn = tf.get_variable('val_b_rnn', initializer = par['val_b_rnn0'], trainable=True)
        # Value network output weights
        with tf.variable_scope('val_output', reuse=tf.AUTO_REUSE):
            val_W_out = tf.get_variable('val_W_out', initializer = par['val_w_out0'], trainable=True)
            val_b_out = tf.get_variable('val_b_out', initializer = par['val_b_out0'], trainable=True)

        task.run_trials(self.policy_hidden_init, self.value_hidden_init, stimulus, truth)

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
        Calculate the loss function dependent on the policy netwokrk parameters
        Equation (2) from Song et al., 2017
        '''
        pol_out = tf.nn.softmax(tf.stack(task.pol_out_history), 1)   # Output of the policy network, a small amount added so log wouldn't get nan
        #pol_out = tf.stack(task.pol_out_history)
        NT = tf.stop_gradient(tf.reduce_sum(task.time_mask))   # Total # of included time points
        # Calculate J (equation 22 bur also baseline will be subtracted):
        # 1) Discard reward at time points that are to be excluded
        #reward = tf.multiply(task.reward, task.time_mask)
        external_reward = tf.stop_gradient(tf.multiply(task.reward, task.time_mask))    # This is the reward value given by the environment
        self.external_reward = external_reward

        time_mask = tf.stop_gradient(task.time_mask)

        # 2) Apply discount (Page 17, Song et al., 2017)
        baseline = tf.stop_gradient(tf.multiply(task.val_out_history, time_mask))
        self.baseline = baseline

        # Calculate discounted future reward per Song et al.,2 017
        Mcausal = np.zeros((par['num_time_steps'], par['num_time_steps']))
        for i in range(par['num_time_steps']):
            # Mcausal[i,i:] = 1 # No discount version
            Mcausal[i,i:] = np.exp(-np.arange(par['num_time_steps']-i)/(par['discount_time_constant']//par['dt']))   # Add discount, 100ms (10 steps) works
        #pdb.set_trace()
        advantage = tf.matmul(Mcausal, external_reward) - baseline

        '''
        # Advantage based on Nick and Greg's code
        Vt = baseline[:-1, :]   # Vt will have all baseline values but the last one
        Vtnext = baseline[1:, :]    # Vt+1 will have all baseline values but the first one
        advantage = external_reward[:-1, :] + par['discount_coef']*Vtnext - Vt
        '''
        self.advantage = advantage
        action_array = tf.stop_gradient(task.action_array)
        # 3) Multiply reward by logpi to get the first term in J (i.e. reward portion)
        logpi = tf.multiply(pol_out, action_array)
        logpi = tf.log(tf.reduce_sum(logpi, axis=1)+1e-7) #tf.log(tf.reduce_sum(logpi, axis=0))
        # logpi = logpi[:-1]     # Discard last time point for some formulations
        task.logpi = logpi
        self.Jn = tf.reduce_sum(tf.multiply(advantage, task.logpi))/(NT - 1)
        #self.Jn = -tf.square(tf.stack(task.pol_out_history) - target)
        # Average Jn values to get average of J
        self.J = tf.reduce_mean(self.Jn)
        # 7) Calculate average regularization term (mentioned as Omega in equation 2)
        with tf.variable_scope('pol_rnn_cell', reuse=True):
            pol_W_rnn = tf.get_variable('pol_W_rnn', dtype=tf.float64)
        # Second norm of the recurrent weight loss, encourages sparse weights
        self.weight_loss_pol = par['weight_cost'] * tf.norm(pol_W_rnn, ord=2) / par['batch_train_size']
        # Sum of firing rates squared (Second norm of their activity matrix), encourages sparse activity
        self.spike_loss_pol = par['spike_cost'] * tf.reduce_mean(tf.reduce_mean(tf.square(tf.stack(task.pol_r_history)), axis=2))
        self.Omega_pol = 0*self.weight_loss_pol + self.spike_loss_pol

        # Caclulate entropy
        #pdb.set_trace()
        log_pol_out = tf.log(pol_out + 1e-7)               # Log of output of the policy network
        # Multiply output and its log
        entropy = tf.multiply(pol_out, log_pol_out) #size: Ntime x 3 x Nbatch size
        # Sum over all the outputs
        entropy = tf.reduce_sum(entropy, axis=1)    #size: Ntime x Nbatch size
        # Apply time mask
        entropy = tf.multiply(entropy, time_mask)
        # Sum across time
        entropy = tf.reduce_sum(entropy, axis=0)    #size: Nbatch size
        # Average across trials
        entropy = -1*tf.reduce_sum(entropy)/NT
        self.entropy = entropy
        self.ent_pol_out = pol_out
        self.ent_log_pol_out = log_pol_out
        self.NT = NT
        # 8) Calculate the loss function for policy network (Equation 2)
        self.Loss_pol = -self.J + self.Omega_pol #- 0.00*self.entropy

        '''
        Calculate the loss function dependent on the value netwokrk parameters
        Equation (4) from Song et al., 2017
        '''

        # 1) Calculate En (Equation 5)
        # Sum of squared of differences averaged across all time points
        # Applt the time mask to output of the value network
        val_out = tf.multiply(tf.stack(task.val_out_history), time_mask)
        # E will minimzie advantage, except instead of Vt, which is not differentiable, we use val_out which is the differentiable variable
        #self.En = tf.square(external_reward[:-1, :] + par['discount_coef']*Vtnext - val_out[:-1, :])
        self.En = tf.square(val_out - advantage)
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
        val_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate']/10)
        """
        Define optimizer, calculate and gradient the the policy network
        """
        pol_opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.pol_grads_and_vars = pol_opt.compute_gradients(self.Loss_pol, var_list = pol_list)
        self.val_grads_and_vars = val_opt.compute_gradients(self.Loss_val, var_list = val_list)
        self.pol_train_op = pol_opt.minimize(self.Loss_pol, var_list = pol_list)
        self.val_train_op = val_opt.minimize(self.Loss_val, var_list = val_list)

def main(task, gpu_id):
    # If a gpu id has been provided, use it
    if gpu_id is not None:
        # Specify the gpu id to be used
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    # Tensorflow finds the supported CPU and GPU devices you can use
    config = tf.ConfigProto()

    trial_length = par['num_time_steps']

    # Calculate shape of the stimulus for this task
    # Define a placeholder for the stimulus the agent sees
    stimulus = tf.placeholder(tf.float64, shape=[trial_length, task.num_inputs, par['batch_train_size']])
    # Define a placeholder for the truth or correct answer about each trial
    truth = tf.placeholder(tf.float64, shape=par['batch_train_size'])
    # A TEMPORARY placeholder for target
    target = tf.placeholder(tf.float64, shape=[trial_length, 3, par['batch_train_size']])

    # Create a model for the given task object
    M = Model()
    # Build the tf structure that runs trials
    M.run_model(task, stimulus, truth, target)
    M.optimize(task, target)
    # Create a model from Nick's code
    stim = stm.Stimulus()
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        for it in range(par['num_iterations']):
            # Create a batch of stimuli, stores in attribute stimulus for the task
            trial_info = stim.generate_trial()
            """
            Run the model
            """
            this_truth = np.zeros(par['batch_train_size'])
            this_truth[trial_info['desired_output'][1,-1,:]==1] = 1   # Match trials
            this_truth[trial_info['desired_output'][2,-1,:]==1] = 2   # Non_match trials
            _, _, vloss, ploss, pol_out, reward, this_action_array, this_time_mask, baseline, advantage = \
                sess.run([M.pol_train_op, M.val_train_op, M.Loss_val, M.Loss_pol, task.pol_out_history, task.reward, task.action_array, \
                task.time_mask, task.val_out_history, M.advantage], \
                {stimulus: np.swapaxes(trial_info['neural_input'], 1, 0), truth: this_truth, target: np.swapaxes(trial_info['desired_output'], 1, 0)})
            reward = reward*this_time_mask; baseline = np.array(baseline)*this_time_mask; pol_out = np.array(pol_out)

            if it%100==0:
                plt.subplot(2, 2, 1)
                plt.plot(pol_out[:,:,0])
                plt.legend(['Fixate', 'match', 'Non-match'])
                if this_truth[0]==1:
                    plt.title('match'+'__'+str(reward.sum()))
                elif this_truth[0]==2:
                    plt.title('non-match'+'__'+str(reward.sum()))
                plt.subplot(2, 2, 2)
                temp = np.tile(this_time_mask[:,0].transpose(), [3, 1]).transpose()*this_action_array[:,:,0]
                plt.plot(temp)
                #plt.plot(advantage[:,0])
                #plt.plot(Pan_results['ploss'].transpose())
                #plt.plot(Pan_results['vloss'].transpose())
                #plt.legend(['-J', 'E'])
                plt.subplot(2, 2, 3)
                plt.plot(np.array(baseline)[:,0])
                #plt.plot(my_reward_cum[:,0])
                #pdb.set_trace()
                plt.plot(reward[:,0])
                plt.title('baseline')
                plt.plot(advantage[:,0])
                plt.subplot(2, 2, 4)
                #plt.plot(np.squeeze(np.array(val_out))[:,0])
                #plt.title('his baseline')
                #plt.plot(trial_info['neural_input'][:,:,0].transpose())
                plt.plot(np.swapaxes(trial_info['desired_output'], 1, 0)[:,:,0])
                plt.title('Neural input')
                plt.savefig(par['save_path']+'Iteration_'+str(it)+'.png')   # save the figure to file
                plt.close()
                print('%5d  |   Vloss: %6.6f   |   Reward: %4d' % (it, vloss, reward.sum()))
