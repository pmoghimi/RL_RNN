"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import analysis
import matplotlib.pyplot as plt
from parameters import *
import os, sys
print(tf.__version__)
import pdb

#print(help(tf))

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#print('Using EI Network:\t', par['EI'])
#print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, actual_reward, pred_reward, actual_action, mask, trial_mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.pred_reward = tf.unstack(pred_reward, axis=0)
        self.actual_action = tf.unstack(actual_action, axis=0)
        self.actual_reward = tf.unstack(actual_reward, axis=0)

        self.time_mask = tf.unstack(mask, axis=0)
        self.trial_mask = tf.unstack(trial_mask, axis=0)    # Added by Pantea

        # Load the initial hidden state activity to be used at the start of each trial
        #self.hidden_init = tf.constant(par['policy_h_init'])
        self.hidden_init = tf.constant(par['h_init'])
        par['pol_n_hidden'] = par['n_hidden']
        par['val_n_hidden'] = par['n_hidden']
        par['pol_num_input'] = par['n_input']
        par['pol_n_output'] = par['n_output']

        # Build the TensorFlow graph
        self.rnn_cell_loop(self.hidden_init, self.hidden_init)

        # Train the model
        self.optimize()


    def rnn_cell_loop(self, h_pol, h_val):


        #self.W_ei = tf.constant(par['EI_matrix'])

        self.h_pol = []
        self.h_val = []
        self.action = []
        self.pol_out = []
        self.val_out = []
        self.mask = []
        self.reward = []
        mask = tf.constant(np.ones((1,par['batch_train_size']), dtype = np.float64))
        self.mask.append(mask)


        """
        Initialize weights and biases
        """
        c = 0.1
        with tf.variable_scope('policy'):
            W_pol_in = tf.get_variable('W_pol_in', initializer = par['pol_w_in0'])
            #rnn0 = np.items(0.2*np.eye(par['n_hidden']))
            W_pol_rnn = tf.get_variable('W_pol_rnn', initializer = par['pol_w_rnn0'])
            b_pol_rnn = tf.get_variable('b_pol_rnn', initializer = par['pol_b_rnn0'])
            W_pol_out = tf.get_variable('W_pol_out', initializer = par['pol_w_out0'])
            b0 = np.zeros([par['pol_n_output'], 1], dtype = np.float64)
            b_pol_out = tf.get_variable('b_pol_out', initializer = par['pol_b_out0'])
        with tf.variable_scope('value'):
            W_val_in = tf.get_variable('W_val_in', initializer = par['val_w_in0'][:,:-3])
            W_val_in_act = tf.get_variable('W_val_in_act', initializer = par['val_w_in0'][:,-3:])
            W_val_rnn = tf.get_variable('W_val_rnn', initializer = par['val_w_rnn0'])
            b_val_rnn = tf.get_variable('b_val_rnn', initializer = par['val_b_rnn0'])
            W_val_out = tf.get_variable('W_val_out', initializer = par['val_w_out0'])
            b_val_out = tf.get_variable('b_val_out', initializer  = par['val_b_out0'])

        #self.syn_x_hist = []
        #self.syn_u_hist = []

        """
        Loop through the neural inputs to the RNN, indexed in time
        """

        self.term1 = []; self.term2 = []; self.term3 = []; self.term4 = []
        for rnn_input, target, time_mask in zip(self.input_data, self.target_data, self.time_mask):
            h_pol, h_val, act, pol_out, val_out, mask, reward  = self.rnn_cell(rnn_input, h_pol, h_val, target, mask, time_mask)
            self.h_pol.append(h_pol)
            self.h_val.append(h_val)
            self.action.append(act)
            self.pol_out.append(pol_out)
            self.val_out.append(val_out)
            self.mask.append(mask)
            self.reward.append(reward)

        self.mask = self.mask[:-1]


    def rnn_cell(self, rnn_input, h_pol, h_val, target, mask, time_mask):

        with tf.variable_scope('policy', reuse = True):
            W_pol_in = tf.get_variable('W_pol_in', dtype=tf.float64)
            W_pol_rnn = tf.get_variable('W_pol_rnn', dtype=tf.float64)
            b_pol_rnn = tf.get_variable('b_pol_rnn', dtype=tf.float64)
            W_pol_out = tf.get_variable('W_pol_out', dtype=tf.float64)
            b_pol_out = tf.get_variable('b_pol_out', dtype=tf.float64)
        with tf.variable_scope('value', reuse = True):
            W_val_in = tf.get_variable('W_val_in', dtype=tf.float64)
            W_val_in_act = tf.get_variable('W_val_in_act', dtype=tf.float64)
            W_val_rnn = tf.get_variable('W_val_rnn', dtype=tf.float64)
            b_val_rnn = tf.get_variable('b_val_rnn', dtype=tf.float64)
            W_val_out = tf.get_variable('W_val_out', dtype=tf.float64)
            b_val_out = tf.get_variable('b_val_out', dtype=tf.float64)

        target_fix = tf.slice(target, [0,0], [1, par['batch_train_size']])
        target_match = tf.slice(target, [1,0], [1, par['batch_train_size']])
        target_non_match = tf.slice(target, [2,0], [1, par['batch_train_size']])
        #print('target_fix', target_fix)
        #print('target_match', target_match)
        #print('target_non_match', target_non_match)

        pol_noise = tf.random_normal([par['pol_n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float64)
        val_noise = tf.random_normal([par['val_n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float64)


        h_pol = tf.nn.relu(h_pol*(1-par['alpha_neuron']) + par['alpha_neuron']*(tf.matmul(W_pol_in, rnn_input) + \
            tf.matmul(W_pol_rnn, h_pol) + b_pol_rnn) + pol_noise)

        pol_out = tf.matmul(W_pol_out, h_pol) + b_pol_out
        #print('pol_out', pol_out)
        action_index = tf.multinomial(tf.transpose(pol_out), 1, seed=0)
        action = tf.cast(tf.transpose(tf.one_hot(tf.squeeze(action_index), par['pol_n_output'])), dtype=tf.float64)
        action_fixate = tf.cast(tf.slice(action, [0,0], [1, par['batch_train_size']]), dtype=tf.float64)
        action_match = tf.cast(tf.slice(action, [1,0], [1, par['batch_train_size']]), dtype=tf.float64)
        action_non_match = tf.cast(tf.slice(action, [2,0], [1, par['batch_train_size']]), dtype=tf.float64)


        # continue_trial is:
        # 1 for trials when action is fixating when its's supposed to fixate
        # 1 for trials when action is fixation when it's supposed to make a decision
        continue_trial = target_fix*action_fixate + (target_match+target_non_match)*action_fixate + (1-time_mask)   #time_mask is always one (the given input), this should just be action_fixate

        reward = mask*time_mask*(target_match*action_match + target_non_match*action_non_match - target_fix*(1-action_fixate))
        mask *= continue_trial
        mask *= (1. - reward)


        term2 = tf.nn.relu(par['alpha_neuron']*(tf.matmul(W_val_rnn, h_val)+tf.matmul(W_val_in, h_pol)+tf.matmul(W_val_in_act, action)+ b_val_rnn))
        #par['alpha_neuron']*(tf.matmul(W_val_in, h_pol) )#+ \
            #tf.matmul(W_val_in_act, action) + tf.matmul(W_val_rnn, h_val) + b_val_rnn) + val_noise;
        self.term2.append(term2)
        h_val = tf.nn.relu(h_val*(1-par['alpha_neuron']) + par['alpha_neuron']*(tf.matmul(W_val_in, h_pol) + \
            tf.matmul(W_val_in_act, action) + tf.matmul(W_val_rnn, h_val) + b_val_rnn) + val_noise)

        val_out = tf.matmul(W_val_out, h_val) + b_val_out   # Was h_pol in Nick's original code

        term1 = tf.matmul(W_val_in, h_pol); self.term1.append(term1)

        term3 = h_val; self.term3.append(term3)
        #self.term3.append(tf.matmul(W_val_in, h_pol))
        #self.term4.append(tf.matmul(W_val_in_act, action))
        term4=tf.matmul(W_val_out, h_val); self.term4.append(term4)

        return h_pol, h_val, action, pol_out, val_out, mask, reward


    def optimize(self):
        """
        Calculate the loss functions and optimize the weights
        """

        val_vars = [var for var in tf.trainable_variables() if 'value' in var.name]
        pol_vars = [var for var in tf.trainable_variables() if 'policy' in var.name]
        #pdb.set_trace()
        #Z = tf.reduce_sum(tf.stack([tf.reduce_sum(time_mask*mask) for (mask, time_mask) in zip(self.mask, self.time_mask)]))
        Z1 = tf.stack(self.mask)
        Z = tf.reduce_sum(tf.stack(self.trial_mask))
        self.mystery = [tf.reduce_sum(time_mask*mask) for (mask, time_mask) in zip(self.mask, self.time_mask)]
        self.mystery2 = [(time_mask*mask) for (mask, time_mask) in zip(self.mask, self.time_mask)]
        self.Z = Z; self.Z1 = Z1

        pol_out_sm = [tf.nn.softmax(pol_out, dim=0) for pol_out in self.pol_out]
        # Make reward cumulative and add discount_time_constant
        self.actual_reward_cum = tf.stack(self.actual_reward)*self.time_mask    # Apply time mask
        #par['discount_time_constant']
        discount = np.arange(par['num_time_steps'])
        #discount = np.ones(par['num_time_steps'], par['batch_train_size'])
        #self.actual_reward_cum =
        Mcausal = np.zeros((par['num_time_steps'], par['num_time_steps']))
        for i in range(par['num_time_steps']):
            # Mcausal[i,i:] = 1 # No discount version
            Mcausal[i,i:] = np.exp(-np.arange(par['num_time_steps']-i)/par['discount_time_constant'])   # Add discount
        self.actual_reward_cum = tf.unstack(tf.matmul(Mcausal, self.actual_reward_cum))

        self.logpi = [tf.reduce_sum(act*tf.log(1e-7+pol_out), axis = 0) for (act,pol_out) in zip(self.actual_action, pol_out_sm)]   # Added by Pantea for comparison
        # Calculate pol loss with cumulative reward
        self.pol_loss = -1.*tf.reduce_sum(tf.stack([(actual_reward - pred_reward)*time_mask*mask*tf.reduce_sum(act*tf.log(1e-7+pol_out), axis = 0) \
            for (pol_out, val_out, act, mask, time_mask, pred_reward, actual_reward) in zip(pol_out_sm, self.val_out, \
            self.actual_action, self.mask, self.time_mask, self.pred_reward, self.actual_reward_cum)]))/Z
        '''
        self.pol_loss = -1.*tf.reduce_sum(tf.stack([(actual_reward - pred_reward)*time_mask*mask*tf.reduce_sum(act*tf.log(1e-7+pol_out), axis = 0) \
            for (pol_out, val_out, act, mask, time_mask, pred_reward, actual_reward) in zip(pol_out_sm, self.val_out, \
            self.actual_action, self.mask, self.time_mask, self.pred_reward, self.actual_reward)]))/Z
        '''
        self.entropy_loss = -1.*tf.reduce_sum(tf.stack([time_mask*mask*pol_out*tf.log(1e-7+pol_out) \
            for (pol_out, mask, time_mask) in zip(pol_out_sm,self.mask, self.time_mask)]))/Z



        """
        self.pol_loss = 1.*tf.reduce_mean(tf.stack([time_mask*tf.squeeze(mask)*(val_out - future_reward)*tf.log(1e-6+tf.nn.softmax(pol_out, dim=0)) \
                for (pol_out, val_out, mask, time_mask) in zip(self.pol_out, self.val_out, self.mask, self.time_mask)]))
        """
        # Nick's old val loss without the cumulative reward
        #self.val_loss = tf.reduce_mean(tf.stack([tf.squeeze(mask)*tf.square(val_out - actual_reward) \
        #        for (val_out, mask, actual_reward) in zip(self.val_out, self.mask, self.actual_reward)]))
        # Val loss with cumulative reward, modified by Pantea
        self.val_loss = tf.reduce_mean(tf.stack([tf.squeeze(mask)*tf.square(val_out - actual_reward) \
                for (val_out, mask, actual_reward) in zip(self.val_out, self.mask, self.actual_reward_cum)]))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.h_pol]

        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

        #self.loss = self.pol_loss + self.val_loss + self.spike_loss
        self.stacked_mask = tf.stack(self.mask)

        opt_val = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        opt_pol = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

        # Added by Pantea to monitor and compare gradients
        self.pol_grads_and_vars = opt_pol.compute_gradients(self.pol_loss + self.spike_loss - 0.0*self.entropy_loss, var_list = pol_vars)
        self.val_grads_and_vars = opt_val.compute_gradients(self.val_loss, var_list = val_vars)

        # Nick's original code, without gradient clipping
        self.train_pol = opt_pol.minimize(self.pol_loss + self.spike_loss - 0.1*self.entropy_loss, var_list = pol_vars)
        self.train_val = opt_val.minimize(self.val_loss, var_list = val_vars)

        # Apply gradiant clipping
        """
        Apply any applicable weights masks to the gradient and clip
        """
        '''
        val_capped_gvs = []
        for grad, var in self.val_grads_and_vars:
            if not str(type(grad)) == "<class 'NoneType'>":
                val_capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        pol_capped_gvs = []
        for grad, var in self.pol_grads_and_vars:
            if not str(type(grad)) == "<class 'NoneType'>":
                pol_capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_val = opt_val.apply_gradients(val_capped_gvs)
        self.train_pol = opt_pol.apply_gradients(pol_capped_gvs)
        '''

def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus()

    n_input, n_hidden, n_output = par['shape']
    N = par['batch_train_size'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float64, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float64, shape=[n_input, par['num_time_steps'], par['batch_train_size']])  # input data
    target = tf.placeholder(tf.float64, shape=[par['n_output'], par['num_time_steps'], par['batch_train_size']])  # input data
    actual_reward = tf.placeholder(tf.float64, shape=[par['num_time_steps'],par['batch_train_size']])
    pred_reward = tf.placeholder(tf.float64, shape=[par['num_time_steps'], par['batch_train_size']])
    actual_action = tf.placeholder(tf.float64, shape=[par['num_time_steps'], par['n_output'], par['batch_train_size']])

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=config) as sess:

        if gpu_id is not None:
            model = Model(x, target, actual_reward, pred_reward, actual_action, mask)
        else:
            #with tf.device("/gpu:0"):
            model = Model(x, target, actual_reward, pred_reward, actual_action,mask)
        init = tf.global_variables_initializer()
        sess.run(init)

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'trial': []}

        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            trial_info = stim.generate_trial()

            """
            Run the model
            """
            pol_out, val_out, pol_rnn, action, stacked_mask, reward = sess.run([model.pol_out, model.val_out, model.h_pol, model.action, \
                 model.stacked_mask,model.reward], {x: trial_info['neural_input'], target: trial_info['desired_output'], mask: trial_info['train_mask']})

            trial_reward = np.squeeze(np.stack(reward))
            trial_action = np.stack(action)
            #plt.imshow(np.squeeze(trial_reward))
            #plt.colorbar()
            #plt.show()

            _, _, pol_loss, val_loss = sess.run([model.train_pol, model.train_val, model.pol_loss, model.val_loss], \
                {x: trial_info['neural_input'], target: trial_info['desired_output'], mask: trial_info['train_mask'], \
                actual_reward: trial_reward, pred_reward: np.squeeze(val_out), actual_action:trial_action })


            accuracy, _, _ = analysis.get_perf(trial_info['desired_output'], action, trial_info['train_mask'])

            #model_performance = append_model_performance(model_performance, accuracy, val_loss, pol_loss, spike_loss, (i+1)*N)

            """
            Save the network model and output model performance to screen
            """
            if i%par['iters_between_outputs']==0 and i > 0:
                print_results(i, N, pol_loss, 0., pol_rnn, accuracy)
                r = np.squeeze(np.sum(np.stack(trial_reward),axis=0))
                print('Mean mask' , np.mean(stacked_mask), ' val loss ', val_loss, ' reward ', np.mean(r), np.max(r))
                #plt.imshow(np.squeeze(stacked_mask[:,:]))
                #plt.colorbar()
                #plt.show()
                #plt.imshow(np.squeeze(trial_reward))
                #plt.colorbar()
                #plt.show()


        """
        Save model, analyze the network model and save the results
        """
        #save_path = saver.save(sess, par['save_dir'] + par['ckpt_save_fn'])
        if par['analyze_model']:
            weights = eval_weights()
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = True, lesion = False, tuning = False, decoding = False, load_previous_file = False, save_raw_data = False)

            # Generate another batch of trials with test_mode = True (sample and test stimuli
            # are independently drawn), and then perform tuning and decoding analysis
            trial_info = stim.generate_trial(test_mode = True)
            y_hat, state_hist, syn_x_hist, syn_u_hist = \
                sess.run([model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = False, lesion = False, tuning = par['analyze_tuning'], decoding = True, load_previous_file = True, save_raw_data = False)



def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, trial_num):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['trial'].append(trial_num)

    return model_performance

def eval_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn.eval(),
        'w_out' : W_out.eval(),
        'b_rnn' : b_rnn.eval(),
        'b_out'  : b_out.eval()
    }

    return weights

def print_results(iter_num, trials_per_iter, perf_loss, spike_loss, state_hist, accuracy):

    print('Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Mean activity {:0.4f}'.format(np.mean(state_hist)))
