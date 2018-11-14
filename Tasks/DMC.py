'''
Delayed Match to Category task
'''
import numpy as np
from parameters import *
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
from TaskTools import *

class dmc:
    def __init__(self, num_trials):
        # Update the "par" variable that contains parameters according to task needs
        self.update_params()
        # With the new parameters update the depndencies (function in parameters.py)
        #update_dependencies()

        self.num_trials = num_trials # Number of trials (train batch size)

        # Convert time parameters from ms to number of steps
        self.fix_dur = par['fix_time'] // par['dt']          # Fixaion priod duration in number of steps
        self.sample_dur = par['sample_time'] // par['dt']    # Sample stimulus period duration in number of steps
        self.delay_dur = par['delay_time'] // par['dt']      # Delay priod duration in number of steps
        self.test_dur = par['test_time'] // par['dt']        # Test stimulus duration in number of steps
        self.resp_dur = par['resp_time'] // par['dt']        # Response period duration in number of steps
        # Total trial duration in number of steps
        self.total_dur = self.fix_dur + self.sample_dur + self.delay_dur + self.test_dur + self.resp_dur

        # Numer of stimulus inputs for this task
        self.num_inputs = par['pol_num_input']

    def run_trials(self, pol_x, val_x, stimulus, target):
        # Unstack input data across the time dimension
        input_data = tf.unstack(stimulus, axis=0)
        # Array to keep history of hidden units activity (x=current, r=firing rate) of the policy network
        self.pol_x_history = []; self.pol_r_history = []; self.pol_out_history = [];
        self.value_input = []; self.val_x_history = []; self.val_r_history = []; self.val_out_history = []
        ############# Decision and reward array initiation##########
        self.action_array = []  # selected actions (stored as one hot)
        self.action = []
        self.reward = [];
        self.time_mask = []
        # Go over input at each point in time (this_u)
        t = 0   # Variable to keep track of time steps
        for this_u in input_data:
            '''
            Calcuate activity and output of policy network
            '''
            pol_x, pol_r, pol_out, other_params = pol_cell(this_u, pol_x)
            # Append current activity and output of the policy network units to their history
            self.pol_x_history.append(tf.transpose(pol_x))
            self.pol_r_history.append(tf.transpose(pol_r))
            self.pol_out_history.append(pol_out)


            """
            Choose action
            Given the output of the policy network, which specifies probabilities, choose an action
            """
            # The multinomial will generate a number in [0, 2] range, 0=fixation, 1=match, 2=nonmatch
            #this_action = tf.multinomial(tf.transpose(pol_out), 1) #tf.multinomial(tf.log(tf.transpose(pol_out)), 1)  # Do NOT remove the log!, or will produce samples not from the given distribution!
            prob = tf.log(tf.nn.softmax(pol_out)) + 1e-10
            pdb.set_trace()
            this_action = tf.multinomial(tf.transpose(prob), 1)

            if t==0:
                self.p = tf.transpose(pol_out)
                self.test = tf.multinomial(self.p, 10000)

            self.action.append(this_action)
            action_array = tf.one_hot(tf.squeeze(this_action), 3)
            action_array = tf.squeeze(tf.cast(action_array, dtype=tf.float32))
            # action_array = tf.transpose(target[t,:,:])    # Ideal agent
            self.action_array.append(tf.transpose(action_array))
            """
            Calculate reward
            """
            if t<(self.total_dur-self.resp_dur-self.test_dur):   # During any period other than response period, action must be 0 (i.e. fixation)
                this_reward = tf.reduce_sum(tf.transpose(action_array)*target[t, :,:], axis=0) - 1#tf.cast(tf.equal(fixate, this_action), dtype=tf.float32) - 1
            elif t>=(self.total_dur-self.resp_dur-self.test_dur):    # During response period, action must be chosen as match or non-match to get a reward
                #this_reward = tf.cast(tf.equal(self.trial_info['match'], this_action), dtype=tf.float32) #- tf.cast(tf.equal(3-truth, this_action), dtype=tf.float32)
                this_reward = tf.reduce_sum(tf.transpose(action_array)*target[t, :,:], axis=0)
            else:
                print('Unknown time value!')
            if t<self.total_dur:
                # Append the current reward to reward history
                self.reward.append(this_reward)
            """
            Update time mask
            """
            if t==0:    # Always include time point 0
                this_mask = tf.constant(np.ones((par['batch_train_size'], 1)), dtype=tf.bool)
                self.time_mask.append(this_mask)
            # Next time point is only included if current action is fixation, i.e. has an entry one for the first column of action_array
            this_mask = tf.logical_and(tf.cast(tf.expand_dims(action_array[:, 0], axis=1), dtype=tf.bool), this_mask)  # Next time point is only included if current action is fixation, i.e. has an entry one for the first column of action_array
            self.time_mask.append(this_mask)

            '''
            Calcuate activity and output of policy network
            '''
            # Concatenate the actions (stored in self.value_nput) with activity of the policy netwrok units
            activity_input = pol_r   # prepare the activity array for concatenation
            action_input = tf.transpose(tf.cast(action_array, dtype=tf.float32))   # Prepare the action array for concatenation
            value_input = tf.concat([activity_input, action_input], axis=0)    # Concatenate the two along the zeroth axis
            # Calculate activity of hidden unit
            val_x, val_r, val_out, other_params = val_cell(activity_input, action_input, val_x)
            # Append activity and output of the value network to their history
            self.val_x_history.append(tf.transpose(val_x))
            self.val_r_history.append(tf.transpose(val_r))
            self.value_input.append(value_input)
            self.val_out_history.append(tf.squeeze(val_out))
            t = t + 1   # Increment time point

        # Reshape the action, reward, and logpi arrays to # time points x # batches (trials)
        self.reward = tf.squeeze(tf.stack(self.reward)) #tf.reshape(self.reward, [self.total_dur, par['batch_train_size']])
        self.action_array = tf.stack(self.action_array)
        self.time_mask = tf.cast(tf.squeeze(tf.stack(self.time_mask)), dtype=tf.float32);
        self.time_mask = self.time_mask[:-1,:]  # Exclude the last time point

    def generate_trial(self):
        '''
        Generates different trial types, including inputs and MT neural responses to this_time_mask
        '''
        # generate tuning functions
        self.motion_tuning = self.create_tuning_functions()
        # end of trial epochs
        eodead = par['dead_time']//par['dt']
        eof = (par['dead_time']+par['fix_time'])//par['dt']
        eos = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        eod = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

        trial_info = {'desired_output'  :  np.zeros((par['pol_n_output'], par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(0, par['noise_in'], size=(par['pol_num_input'], par['num_time_steps'], par['batch_train_size']))}

        # set to mask equal to zero during the dead time
        trial_info['train_mask'][:eodead, :] = 0
        for t in range(par['batch_train_size']):
            """
            Generate trial paramaters
            """
            sample_dir = np.random.randint(par['num_motion_dirs'])
            match = np.random.randint(2)
            # categorize between two equal size, contiguous zones
            sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
            if match == 1: # match trial
                # do not use sample_dir as a match test stimulus
                dir0 = int(sample_cat*par['num_motion_dirs']//2)
                dir1 = int(par['num_motion_dirs']//2 + sample_cat*par['num_motion_dirs']//2)
                possible_dirs = list(range(dir0, dir1))
                test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
            else:
                test_dir = sample_cat*(par['num_motion_dirs']//2) + np.random.randint(par['num_motion_dirs']//2)
                test_dir = np.int_((test_dir+par['num_motion_dirs']//2)%par['num_motion_dirs'])
            """
            Calculate neural input based on sample and tests
            """
            # SAMPLE stimulus
            trial_info['neural_input'][:, eof:eos, t] += np.reshape(self.motion_tuning[:,sample_dir],(-1,1))
            # TEST stimulus
            trial_info['neural_input'][:, eod:, t] += np.reshape(self.motion_tuning[:,test_dir],(-1,1))
            """
            Determine the desired network output response
            """
            trial_info['desired_output'][0, eodead:eod, t] = 1  # Fixation output
            trial_info['train_mask'][ eod:, t] = 1 # can use a greater weight for test period if needed
            if match == 1:
                trial_info['desired_output'][1, eod:, t] = 1    # Match output
            else:
                trial_info['desired_output'][2, eod:, t] = 1    # Non-match output
            """
            Append trial info
            """
            trial_info['sample'][t] = sample_dir
            trial_info['test'][t] = test_dir
            trial_info['match'][t] = match
        self.trial_info = trial_info

    def create_tuning_functions(self):
        """
        Generate tuning functions for the task
        """
        motion_tuning = np.zeros((par['num_motion_tuned'], par['num_motion_dirs']))

        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/par['num_motion_tuned']))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs']))

        for n in range(par['num_motion_tuned']):
            for i in range(len(stim_dirs)):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                motion_tuning[n,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])
        return motion_tuning


    def update_params(self):
        # Stimulus parameters
        par['num_motion_dirs'] = 8      # Total number of motion directions
        par['tuning_height'] = 4        # magnitutde scaling factor for von Mises
        par['kappa'] = 2                # concentration scaling factor for von Mises
        # Specify task and network parameters based on task needs
        par['dead_time'] = 0              # No dead period for this task
        par['fix_time'] = 200             # Fixaion priod duration in ms
        par['sample_time'] = 300          # Sample stimulus period duration in ms
        par['delay_time'] = 50            # Delay priod duration in ms
        par['test_time'] = 300            # Test period duration in ms
        par['resp_time'] = 400            # Response period duration in ms
        #par['mask_duration'] = par['test_time']
        # Length of each trial in time steps
        par['num_time_steps'] = np.int32((par['fix_time'] + par['sample_time'] + par['delay_time'] + par['test_time'] + par['resp_time'])/par['dt'])
        par['trial_length'] = par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']+par['resp_time']

        '''
        Policy network
        '''
        # Policy network shape
        par['num_motion_tuned'] = 36                     # Number of input MT neurons
        par['pol_num_input'] = par['num_motion_tuned']   # Number of inputs to the policy network

        par['pol_n_output'] = 3     # Number of outputs for the policy network
        # Initial hidden unit activity for the policy network
        par['policy_h_init'] = 0.1*np.ones((par['pol_n_hidden'], par['batch_train_size']), dtype=np.float32)

        # Initialize input weights for the policy network
        c = 0.1
        par['pol_w_in0'] = np.float32(np.random.uniform(-c,c, size = [par['pol_n_hidden'], par['pol_num_input']]))
        par['pol_w_rnn0'] = np.float32(np.random.uniform(-c,c, size = [par['pol_n_hidden'], par['pol_n_hidden']]))
        #par['pol_w_rnn0'] = 0.54*np.float32(np.eye(par['pol_n_hidden']))
        par['pol_b_rnn0'] = np.float32(np.random.uniform(-c,c, size = [par['pol_n_hidden'], 1]))
        par['pol_w_out0'] = np.float32(np.random.uniform(-c,c, size = [par['pol_n_output'], par['pol_n_hidden']]))
        par['pol_b_out0'] = np.zeros([par['pol_n_output'], 1], dtype = np.float32)
        # Learning masks for rnn and output Weights
        par['pol_w_rnn_mask'] = np.ones((par['pol_n_hidden'], par['pol_n_hidden']), dtype=np.float32)
        par['pol_w_out_mask'] = np.ones((par['pol_n_output'], par['pol_n_hidden']), dtype=np.float32)
        # Dale's law if applicable
        if par['EI']:
            for i in range(par['pol_n_hidden']):
                # Remove self connections
                par['pol_w_rnn0'][i,i] = 0
                par['pol_w_rnn_mask'] = np.ones((par['pol_n_hidden'], par['pol_n_hidden']), dtype=np.float32) - np.eye(par['pol_n_hidden'])
            ind_inh = np.where(par['pol_EI_list'] == -1)[0]
            par['pol_w_out0'][:, ind_inh] = 0
            par['pol_w_out_mask'][:, ind_inh] = 0
        '''
        Value network
        '''
        # Value network shape
        # Reminder: Value network only has one output: expected reward (Song et al., 2017)
        # Value network gets as input activity of all the hidden units in the policy network
        # as well as agent (i.e. which decisions have zero values (not made) and which one has a value of one (made))
        par['val_num_input'] = par['pol_n_hidden'] + par['pol_n_output']   # Number of inputs to the value network
        par['val_n_output'] = 1    # Number of outputs for the value network
        # Initial hidden unit activity for the policy network
        par['value_h_init'] = 0.1*np.ones((par['val_n_hidden'], par['batch_train_size']), dtype=np.float32)

        # Initialize input weights for the value network
        c = 0.1
        par['val_w_in0'] = np.float32(np.random.uniform(-c,c, size = [par['val_n_hidden'], par['val_num_input']]))
        par['val_w_rnn0'] = np.float32(np.random.uniform(-c,c, size = [par['val_n_hidden'], par['val_n_hidden']]))
        #par['val_w_rnn0'] = 0.54*np.float32(np.eye(par['val_n_hidden']))
        par['val_b_rnn0'] = np.float32(np.random.uniform(-c,c, size = [par['val_n_hidden'], 1]))

        par['val_w_out0'] = np.float32(np.random.uniform(-c,c, size = [par['val_n_output'], par['val_n_hidden']]))
        par['val_b_out0'] = np.float32(np.random.uniform(-c,c, size = [1, 1]))
        # Learning masks for rnn and output Weights
        par['val_w_rnn_mask'] = np.ones((par['val_n_hidden'], par['val_n_hidden']), dtype=np.float32)
        par['val_w_out_mask'] = np.ones((par['val_n_output'], par['val_n_hidden']), dtype=np.float32)
        # Dale's law if applicable
        if par['EI']:
            for i in range(par['pol_n_hidden']):
                # Remove self connections
                par['pol_w_rnn0'][i,i] = 0
                par['val_w_rnn_mask'] = np.ones((par['val_n_hidden'], par['val_n_hidden']), dtype=np.float32) - np.eye(par['pol_n_hidden'])
            ind_inh = np.where(par['val_EI_list'] == -1)[0]
            par['val_w_out0'][:, ind_inh] = 0
            par['val_w_out_mask'][:, ind_inh] = 0
