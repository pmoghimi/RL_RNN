'''
One Interval Categorization task
'''
import numpy as np
from parameters import *
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
from TaskTools import *

class oic:
    def __init__(self, num_trials):
        # Update the par variable that contains parameters according to task needs
        self.update_params()
        # With the new parameters update the depndencies (function in parameters.py)
        update_dependencies()

        self.num_trials = num_trials # Number of trials (train batch size)


        # Convert time parameters from ms to number of steps
        self.fix_dur = par['fix_dur'] // par['dt']          # Fixaion priod duration in number of steps
        self.stim_dur = par['stim_dur'] // par['dt']        # Stimulus period duration in number of steps
        self.resp_dur = par['resp_dur'] // par['dt']        # Response period duration in number of steps
        # Total trial duration in number of steps
        self.total_dur = self.fix_dur + self.stim_dur + self.resp_dur

        # Numer of stimulus inputs for this task
        self.num_inputs = par['pol_num_input']

    def run_trials(self, pol_x, val_x, stimulus, truth):
        # Unstack input data across the time dimension
        input_data = tf.unstack(stimulus, axis=0)
        # Put truth in proper shape
        truth = tf.expand_dims(truth, axis=1)
        # Define tensor flow objects for fixation, so the selected action can be compared to
        fixate = tf.constant(np.zeros((par['batch_train_size'], 1)), dtype=np.float64)
        # Array to keep history of hidden units activity (x=current, r=firing rate) of the policy network
        self.pol_x_history = []; self.pol_r_history = []; self.pol_out_history = []; self.pol_out_history0 = [];
        self.value_input = []; self.val_x_history = []; self.val_r_history = []; self.val_out_history = []
        ############# Decision and reward array initiation##########
        self.actions = [];  self.action_array = []  # Array version of action (binary)
        self.reward = []; self.discount = []  # See page 17 of Song et al., 2017
        self.logpi = []; self.time_mask = []
        self.term1 = []; self.term2 = []; self.term3 = []; self.term4 = []
        # Go over input at each point in time (this_u)
        self.t = 0   # Variable to keep track of time (in # itme points)
        cont_flag = tf.constant(np.ones((self.num_inputs, par['batch_train_size'])), dtype=tf.float64)   # Continuation of trial flag
        for this_u in input_data:
            pol_x, pol_r, other_params = pol_cell(this_u, pol_x)
            # Append current activity of the policy network units to their history
            self.pol_x_history.append(tf.transpose(pol_x))
            self.pol_r_history.append(tf.transpose(pol_r))
            '''
            # 2) Policy network:
            Given the hidden state firing rate at time t, get output at time t (policy)
            '''
            with tf.variable_scope('pol_output', reuse=True):
                pol_W_out = tf.get_variable('pol_W_out', dtype=tf.float64)
                pol_b_out = tf.get_variable('pol_b_out', dtype=tf.float64)
            pol_out_0 = tf.matmul(pol_W_out, pol_r) + pol_b_out   # Linear part, equation 6
            pol_out = pol_out_0 #tf.nn.softmax(pol_out_0, 0)  # Softmax part, equation 7

            # Append current output of the policy network to its history
            self.pol_out_history.append(pol_out)
            self.pol_out_history0.append(pol_out_0)
            '''
            # 3) Choose action
            Given the output of the policy network, which specifies probabilities, choose an action
            '''
            # The multinomial will generate a number in [0, 2] range, 0=fixation, 1=match, 2=nonmatch
            # The multinomial function of tensorflow requires logits, hence the log
            this_action = tf.multinomial(tf.transpose(pol_out), 1, seed=0) #tf.multinomial(tf.log(tf.transpose(pol_out)), 1)  # Do NOT remove the log!, or will produce samples not from the given distribution!
            action_array = tf.one_hot(this_action, 3)
            action_array = tf.squeeze(tf.cast(action_array, dtype=tf.float64))
            this_action = tf.cast(this_action, dtype=tf.float64)
            # I just need to do it as an input to the value network, otherwise, having the actions vector as is for reward calculation is better
            self.actions.append(this_action)
            # 5) Given the selected action for each batch (trial), calculate the state of the system and its reward
            # Update action in array form
            self.action_array.append(action_array)
            # Caclaulte reward
            if self.t<(self.total_dur-self.resp_dur-self.test_dur):   # During any period other than response period, action must be 0 (i.e. fixation)
                this_reward = tf.cast(tf.equal(fixate, this_action), dtype=tf.float64) - 1
            elif self.t>=(self.total_dur-self.resp_dur-self.test_dur):    # During response period, action must be chosen as match or non-match to get a reward
                this_reward = tf.cast(tf.equal(truth, this_action), dtype=tf.float64) # - tf.cast(tf.not_equal(truth, this_action), dtype=tf.float64)
            else:
                print('Unknown time value!')
            # Update time mask
            if self.t==0:    # Always include time point 0
                this_mask = tf.constant(np.ones((par['batch_train_size'], 1)), dtype=tf.bool)
                self.time_mask.append(this_mask)
                this_mask = tf.logical_and(this_mask, tf.equal(fixate, this_action))
            else:   # After time point 0
                self.time_mask.append(this_mask)
                this_mask = tf.logical_and(this_mask, tf.equal(fixate, this_action))
            # Append the current reward to the corresponding arrays
            self.reward.append(this_reward)
            # Update reward discount according to page 17 of Song et al., 2017
            self.discount.append(np.exp(-self.t*par['dt']/par['discount_time_constant']))
            '''
            4) Value network:
            Given activity of policy network units and actions up to current time,
            calculate activity of hidden units in the value network
            x is the input current to each cell and r is be firing rate
            '''
            # Concatenate the actions (stored in self.value_nput) with activity of the policy netwrok units
            activity_input = pol_r   # prepare the activity array for concatenation
            action_input = tf.transpose(tf.cast(action_array, dtype=tf.float64))   # Prepare the action array for concatenation
            value_input = tf.concat([activity_input, action_input], axis=0)    # Concatenate the two along the zeroth axis
            # Calculate activity of hidden unit
            if par['val_unit_type'] == 'STSP':
                val_x, val_r, other_params = val_cell(value_input, val_x, val_syn_u_init=val_syn_u_init, val_syn_x_init=val_syn_x_init)
            else:
                #val_x, val_r, other_params = val_cell(value_input, val_x)
                val_x, val_r, other_params = val_cell(activity_input, action_input, val_x)
                self.term1.append(other_params['term1'])
                self.term2.append(other_params['term2'])
                self.term3.append(other_params['term3'])
                self.term4.append(other_params['term4'])
            # Append current activity of the policy network units to their history
            self.val_x_history.append(tf.transpose(val_x))
            self.val_r_history.append(tf.transpose(val_r))
            self.value_input.append(value_input)
            '''
            5) Value network:
            Given the hidden state activity at time t, get output at time t (predicted reward)
            '''
            with tf.variable_scope('val_output', reuse=True):
                val_W_out = tf.get_variable('val_W_out', dtype=tf.float64)
                val_b_out = tf.get_variable('val_b_out', dtype=tf.float64)
            val_out = tf.matmul(val_W_out, val_r) + val_b_out   # Linear part
            # Append current output of the policy network to its history
            self.val_out_history.append(tf.squeeze(val_out))
            self.t = self.t + 1   # Increment time point

        # Reshape the action, reward, and logpi arrays to # time points x # batches (trials)
        self.reward = tf.squeeze(tf.stack(self.reward)) #tf.reshape(self.reward, [self.total_dur, par['batch_train_size']])
        self.actions = tf.squeeze(tf.stack(self.actions)) #tf.reshape(self.actions, [self.total_dur, par['batch_train_size']])
        self.action_array = tf.stack(self.action_array)
        self.time_mask = tf.squeeze(tf.stack(self.time_mask)); self.time_mask = tf.cast(self.time_mask, dtype=tf.float64)

    def generate_trial(self):
        """
        Generate a pne interval task
        Goal is to determine which category the stimulus belongs to
        """
        trial_length = self.total_dur   # Length of the trial
        # end of trial epochs
        eof = par['fix_dur']//par['dt']     # End of fixation period
        eos = (par['fix_dur']+par['stim_dur'])//par['dt']   # End of stimulation period
        eot = (par['fix_dur']+par['stim_dur']+par['resp_dur'])//par['dt']   # End of trial period


        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']

        trial_info = {'desired_output'  :  np.zeros((par['n_output'], trial_length, par['batch_train_size']),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, par['batch_train_size']),dtype=np.float32),
                      'stim'            :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['n_input'], trial_length, par['batch_train_size']))}

        # Generate tuning curves
        self.motion_tuning = self.create_tuning_functions()

        for t in range(par['batch_train_size']):
            """
            Generate trial paramaters
            """
            stim_dir = np.random.randint(par['num_motion_dirs'])    # Direction of motion

            """
            Generate stimulus
            """
            # categorize between two equal size, contiguous zones
            cat = np.floor(stim_dir/(par['num_motion_dirs']/2))
            """
            Calculate neural input based on stimulus
            """
            pdb.set_trace()
            # stimulus
            trial_info['neural_input'][:, eof:eos, t] += np.reshape(self.motion_tuning[:,stim_dir],(-1,1))
            """
            Determine the desired network output response
            """
            trial_info['desired_output'][0, eodead:eod_current, t] = 1
            if not catch:
                trial_info['train_mask'][ eod_current:, t] = 1 # can use a greater weight for test period if needed
                if match == 0:
                    trial_info['desired_output'][1, eod_current:, t] = 1
                else:
                    trial_info['desired_output'][2, eod_current:, t] = 1
            else:
                trial_info['desired_output'][0, eod_current:, t] = 1


            """
            Append trial info
            """
            trial_info['sample'][t] = sample_dir
            trial_info['test'][t] = test_dir
            trial_info['rule'][t] = rule
            trial_info['catch'][t] = catch
            trial_info['match'][t] = match

        return trial_info

    def create_tuning_functions(self):
        """
        Generate tuning functions for the Postle task
        """
        motion_tuning = np.zeros((par['num_motion_tuned'], par['num_receptive_fields'], par['num_motion_dirs']))

        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields'])))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs']))

        for n in range(par['num_motion_tuned']//par['num_receptive_fields']):
            for i in range(len(stim_dirs)):
                for r in range(par['num_receptive_fields']):
                    d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                    n_ind = n+r*par['num_motion_tuned']//par['num_receptive_fields']
                    motion_tuning[n_ind,r,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])

        return np.squeeze(motion_tuning)


    def update_params(self):
        # Specify task and network parameters based on task needs
        par['fix_dur'] = 200         # Fixaion priod duration in ms
        par['stim_dur'] = 300     # Stimulus persentation period duration in ms
        par['resp_dur'] = 400       # Response period duration in ms
        par['num_time_steps'] = np.int32((par['fix_dur'] + par['stim_dur'] + par['resp_dur'])/par['dt'])
        # Policy network shape
        par['num_motion_tuned'] = 36                     # Number of MT neurons
        par['num_fix_tuned'] = 0                         # NUmber of neurons encoding the fixation dot
        par['pol_num_input'] = par['num_motion_tuned'] + par['num_fix_tuned']    # Number of inputs to the policy network

        # Number of input neurons
        par['n_input'] = par['pol_num_input']
        # Inputs for hte RDM task are: Fixation cue, right evidence, left evidence
        par['pol_n_hidden'] = 100   # Number of hidden units in the recurrent network
        par['n_hidden'] = par['pol_n_hidden']
        par['pol_n_output'] = 3     # Number of outputs for the policy network
        par['n_output'] = par['pol_n_output']
        # General network shape
        par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

        # Initial hidden unit activity for the policy network
        par['policy_h_init'] = par['h_init'] #0.1*np.ones((par['pol_n_hidden'], par['batch_train_size']), dtype=np.float64)
        # Initial hidden unit activity for the value network
        par['value_h_init'] = par['h_init'] #0.1*np.ones((par['val_n_hidden'], par['batch_train_size']), dtype=np.float64)
