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
from TaskTools import *

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#print('Using EI Network:\t', par['EI'])
#print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""

class Model:
    def __init__(self):
        # Initialize hidden unit activity of poicy and value networks
        self.policy_hidden_init = tf.constant(par['policy_h_init'], dtype=np.float32)
        self.value_hidden_init = tf.constant(par['value_h_init'], dtype=np.float32)

    def run_model(self, task, stimulus, target):
        ###### Policy Netwrok########
        with tf.variable_scope('pol_cell', reuse=tf.AUTO_REUSE):
            pol_W_in = tf.get_variable('pol_W_in', initializer = par['pol_w_in0'], trainable=True, dtype=np.float32)
            pol_W_rnn = tf.get_variable('pol_W_rnn', initializer = par['pol_w_rnn0'], trainable=True, dtype=np.float32)
            pol_b_rnn = tf.get_variable('pol_b_rnn', initializer = par['pol_b_rnn0'], trainable=True, dtype=np.float32)
            pol_W_out = tf.get_variable('pol_W_out', initializer = par['pol_w_out0'], trainable=True, dtype=np.float32)
            pol_b_out = tf.get_variable('pol_b_out', initializer = par['pol_b_out0'], trainable=True, dtype=np.float32)

        ###### Value Netwrok########
        with tf.variable_scope('val_cell', reuse=tf.AUTO_REUSE):
            val_W_in_pol = tf.get_variable('val_W_in_pol', initializer = par['val_w_in0'][:,:-3], trainable=True) # Weights to be multiplied by activity pf policy network hidden units
            val_W_in_act = tf.get_variable('val_W_in_act', initializer = par['val_w_in0'][:,-3:], trainable=True) # Weights to be multiplied by selected actions
            val_W_rnn = tf.get_variable('val_W_rnn', initializer = par['val_w_rnn0'], trainable=True)
            val_b_rnn = tf.get_variable('val_b_rnn', initializer = par['val_b_rnn0'], trainable=True)
            val_W_out = tf.get_variable('val_W_out', initializer = par['val_w_out0'], trainable=True)
            val_b_out = tf.get_variable('val_b_out', initializer = par['val_b_out0'], trainable=True)

        # Unstack input data across the time dimension
        input_data = tf.unstack(stimulus, axis=0)
        # Array to keep history of hidden units activity (x=current, r=firing rate) of the policy network
        task.pol_out_history = []; task.val_out_history = []
        ############# Decision and reward array initiation##########
        task.action_array = []; task.reward = []; task.time_mask = []
        # Go over input at each point in time (this_u)
        t = 0   # Variable to keep track of time steps
        pol_x = self.policy_hidden_init; val_x = self.value_hidden_init;
        for this_u in input_data:
            '''
            Calcuate activity and output of policy network
            '''
            pol_r = tf.nn.relu(pol_x)
            pol_rec_noise = tf.random_normal([par['pol_n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32)
            # Caclulate next x
            pol_x = (1-par['alpha_neuron'])*pol_r + \
                        par['alpha_neuron']*(pol_W_rnn@ pol_r + pol_W_in@this_u + pol_b_rnn) + pol_rec_noise
            # Firing rates (r)
            pol_next_r = tf.nn.relu(pol_x)
            # Calculate output
            pol_out = pol_W_out@pol_next_r + pol_b_out
            task.pol_out_history.append(pol_out)
            """
            Choose action
            Given the output of the policy network, which specifies probabilities, choose an action
            """
            # The multinomial will generate a number in [0, 2] range, 0=fixation, 1=match, 2=nonmatch
            this_action = tf.multinomial(tf.transpose(pol_out), 1) #tf.multinomial(tf.log(tf.transpose(pol_out)), 1)  # Do NOT remove the log!, or will produce samples not from the given distribution!
            action_array = tf.one_hot(tf.squeeze(this_action), 3)
            action_array = tf.squeeze(tf.cast(action_array, dtype=tf.float32))
            task.action_array.append(tf.transpose(action_array))
            """
            Calculate reward
            """
            if t<(task.total_dur-task.resp_dur-task.test_dur):   # During any period other than response period, action must be 0 (i.e. fixation)
                this_reward = tf.reduce_sum(tf.transpose(action_array)*target[t, :,:], axis=0) - 1#tf.cast(tf.equal(fixate, this_action), dtype=tf.float32) - 1
            elif t>=(task.total_dur-task.resp_dur-task.test_dur):    # During response period, action must be chosen as match or non-match to get a reward
                this_reward = tf.reduce_sum(tf.transpose(action_array)*target[t, :,:], axis=0)
            else:
                print('Unknown time value!')
            # Append the current reward to reward history
            task.reward.append(this_reward)
            """
            Update time mask
            """
            if t==0:    # Always include time point 0
                this_mask = tf.constant(np.ones((par['batch_train_size'], 1)), dtype=tf.bool)
                task.time_mask.append(this_mask)
            # Next time point is only included if current action is fixation, i.e. has an entry one for the first column of action_array
            this_mask = tf.logical_and(tf.cast(tf.expand_dims(action_array[:, 0], axis=1), dtype=tf.bool), this_mask)  # Next time point is only included if current action is fixation, i.e. has an entry one for the first column of action_array
            task.time_mask.append(this_mask)
            """
            Calcuate activity and output of policy network
            """
            # Concatenate the actions (stored in self.value_nput) with activity of the policy netwrok units
            action_input = tf.transpose(tf.cast(action_array, dtype=tf.float32))   # Prepare the action array for concatenation
            # Calculate activity of hidden unit
            val_r = tf.nn.relu(val_x)
            val_rec_noise = tf.random_normal([par['val_n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32)
            # Caclulate next x
            val_x = (1-par['alpha_neuron'])*val_r + \
                        par['alpha_neuron']*(tf.matmul(val_W_rnn,val_r) + \
                        tf.matmul(val_W_in_pol,pol_r) + tf.matmul(val_W_in_act,action_input) + \
                        val_b_rnn) + val_rec_noise
            # Firing rates (r)
            val_next_r = tf.nn.relu(val_x)
            # Calculate output
            val_out = val_W_out @ val_next_r + val_b_out
            task.val_out_history.append(tf.squeeze(val_out))
            t = t + 1   # Increment time point

        # Reshape the action, reward, and logpi arrays to # time points x # batches (trials)
        task.reward = tf.squeeze(tf.stack(task.reward)) #tf.reshape(self.reward, [self.total_dur, par['batch_train_size']])
        task.action_array = tf.stack(task.action_array)
        task.time_mask = tf.cast(tf.squeeze(tf.stack(task.time_mask)), dtype=tf.float32);
        task.time_mask = task.time_mask[:-1,:]  # Exclude the last time point

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
            advantage = Mcausal@external_reward - baseline
            self.advantage = advantage
        elif par['learning_rule'] == 'TD':
            # Advantage based on Nick and Greg's code
            Vt = baseline[:-1, :]   # Vt will have all baseline values but the last one
            Vtnext = baseline[1:, :]    # Vt+1 will have all baseline values but the first one
            advantage = external_reward[:-1, :] + par['discount_coef']*Vtnext - Vt
            self.advantage = advantage
        elif par['learning_rule'] == 'Supervised':
            self.advantage = external_reward  # Some random assignemnt for the sake of consistency

        if par['learning_rule'] == 'diff' or par['learning_rule'] == 'TD':
            Jn = tf.reduce_sum(advantage*logpi, axis = 0)
        elif par['learning_rule'] == 'Supervised':
            Jn = -tf.square(tf.stack(task.pol_out_history) - target)
        else:
            disp('Unknown learning rule')
        # Average Jn values to get average of J
        self.J = tf.reduce_mean(Jn)

        log_pol_out = tf.log(tf.stack(task.pol_out_history) + 1e-7)               # Log of output of the policy network
        # Multiply output and its log
        entropy = tf.multiply(tf.stack(task.pol_out_history), log_pol_out) #size: Ntime x 3 x Nbatch size
        # Sum over all the outputs
        entropy = tf.reduce_sum(entropy, axis=1)    #size: Ntime x Nbatch size
        # Apply time mask
        entropy = tf.multiply(entropy, tf.stop_gradient(task.time_mask))
        # Sum across time
        entropy = tf.reduce_sum(entropy, axis=0)    #size: Nbatch size
        # Average across trials
        entropy = -1*tf.reduce_sum(entropy)
        self.entropy = entropy

        # Calculate the loss function for policy network
        self.Loss_pol = -self.J + 0.0*self.entropy

        '''
        Calculate the loss function for baseline network
        '''
        val_out = tf.multiply(tf.stack(task.val_out_history), tf.stop_gradient(task.time_mask))
        if par['learning_rule'] == 'TD':
            # E will minimzie advantage, except instead of Vt, which is not differentiable, we use val_out which is the differentiable variable
            En = tf.square(external_reward[:-1, :] + par['discount_coef']*Vtnext - val_out[:-1, :])
        elif par['learning_rule'] == 'diff':
            En = tf.square(val_out - Mcausal@external_reward)  # We are adding the baseline back, which was subtracted for calculating J
        elif par['learning_rule'] == 'Supervised':
            En = -tf.reduce_mean(val_out)   # Some arbitrary thing to get the code work for all learning rules
        else:
            disp('Unknown learning rule')
        # Average En values to get E
        self.E = tf.reduce_mean(En)
        # Calculate loss for the value network (Equation 4)
        self.Loss_val = self.E

        self.Loss = self.Loss_pol + self.Loss_val
        """
        Define optimizer, calculate gradients for both networks
        """
        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.train_op = opt.minimize(self.Loss)

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
    stimulus = tf.placeholder(tf.float32, shape=[par['num_time_steps'], task.num_inputs, par['batch_train_size']])
    # Define a placeholder for the target or truth
    target = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['pol_n_output'], par['batch_train_size']])

    # Create a model for the given task object
    M = Model()
    #M = model_works.Model()
    # Build the tf structure that runs trials
    M.run_model(task, stimulus, target)
    M.optimize(task, target)

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        for it in range(par['num_iterations']):
            # Create a batch of stimuli, stores in attribute stimulus for the task
            task.generate_trial()
            trial_info = task.trial_info
            """
            Run the model
            """
            _, loss, pol_out, reward, this_action_array, this_time_mask, baseline, advantage, logpi, pol_out_final = \
                sess.run([M.train_op, M.Loss, task.pol_out_history, M.external_reward, task.action_array, \
                task.time_mask, M.baseline, M.advantage, M.logpi, M.pol_out], \
                {stimulus: np.swapaxes(trial_info['neural_input'], 1, 0), target:np.swapaxes(trial_info['desired_output'], 1, 0)})
            pol_out = np.array(pol_out);
            if it%par['iters_between_outputs']==0:
                #pdb.set_trace()
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
                plt.plot(advantage[:,0])
                plt.title('baseline')
                plt.subplot(2, 2, 4)
                plt.plot(trial_info['neural_input'][:,:,0].transpose())
                plt.savefig(par['save_path']+'Iteration_'+str(it)+'.png')   # save the figure to file
                plt.close()
                print('%5d  |   Loss: %6.6f   |   Reward: %4d' % (it, loss, reward.sum()))
