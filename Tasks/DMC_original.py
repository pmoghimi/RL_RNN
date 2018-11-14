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
        '''
        Initializes the DMC taks object
        Task is divided to the following epochs:
        1) Fixation period:
            fixation dot on, agent has to Fixate, breaking fixation results in reward = -1
        2) Sample period:
            Sample stimulus is on, fixation dot is on, agent has to Fixate, breaking fixation results in reward = -1
        3) Delay period:
            Sample stimulus is off, fixation dot is on, the agent has to Fixate, breaking fixation results in reward = -1
        4) Test period:
            Test stimulus is on, fixation dot is on, agent has to fixate, breaking fixation results in reward = -1
        5) Response period: Test stimulus is off, fixation dot is off, agent has to break fixation,
           indicate if sample and test stimuli are in the same (match) or different (non-match) categories
           Correct response results in reward = +1
        Inputs:
        Fixation input: Assumes values between 0 (fixation dot off) and 1 (fixation dot off)
        Motion inputs: Several inputs whose values reflect response of a number of motion tuned (MT) neurons to motion direction in each trial
                       These inputs assumea non-zero value when either the sample or test stimuli are on

        Outputs:
            Fixation output: The output that is 0 when agent is fixating and is 0 otherwise
            Match output: The output that is 1 when agent indicates a match between sample and test, 0 otherwise
            Non-match output: The output that is 1 when agent indicates a non-match between sample and test, 0 otherwise
        '''

        # Update the par variable that contains parameters according to task needs
        self.update_params()
        # With the new parameters update the depndencies (function in parameters.py)
        update_dependencies()

        self.num_trials = num_trials # Number of trials (train batch size)


        # Convert time parameters from ms to number of steps
        self.fix_dur = par['fix_dur'] // par['dt']          # Fixaion priod duration in number of steps
        self.sample_dur = par['sample_dur'] // par['dt']    # Sample stimulus period duration in number of steps
        self.delay_dur = par['delay_dur'] // par['dt']      # Delay priod duration in number of steps
        self.test_dur = par['test_dur'] // par['dt']        # Test stimulus duration in number of steps
        self.resp_dur = par['resp_dur'] // par['dt']        # Response period duration in number of steps
        # Total trial duration in number of steps
        self.total_dur = self.fix_dur + self.sample_dur + self.delay_dur + self.test_dur + self.resp_dur

        # Numer of stimulus inputs for this task
        self.num_inputs = par['pol_num_input']

        # Category boundary for the motion direction categories (in degrees)
        self.boundary = 45  # Category boundary between 0 and 180
        self.boundary = np.array([self.boundary, self.boundary+ 180]) # The boundary required two angle values

        self.num_motion_dirs = 8    # Possible number of motion directions to choose from
        possible_dirs = np.arange(0, self.num_motion_dirs)*360/self.num_motion_dirs
        self.possible_dirs = possible_dirs
        # If any of the possible directions are exactly on the boundary, shift the boundary by half of the distance between two consecutive motion directions
        if np.logical_or(np.any(self.boundary[0]==possible_dirs), np.any(self.boundary[1]==possible_dirs)):
            self.boundary = self.boundary + 360/self.num_motion_dirs/2

    def create_batch(self):
        """
        This function creates a batch of trials
        It randomly samples a number of sample and test motion direction pairs from a possible number of ddirections
        """
        # Sample stimulus directions
        sample_dirs = np.random.randint(0, self.num_motion_dirs, (self.num_trials, 1))*360/self.num_motion_dirs
        # Test stimulus directions
        test_dirs = np.random.randint(0, self.num_motion_dirs, (self.num_trials, 1))*360/self.num_motion_dirs
        self.directions = np.concatenate((sample_dirs, test_dirs), axis=1)

    def create_stimulus(self, directions = None):
        """
        Generate tuning functions for the MT neurons and calculate their response to each motion direction for each trial
        """
        # 1) Set up tuning function parameters and get sample and test directions
        self.tuning_height = 4      # Magnitutde scaling factor for von Mises Fisher function
        self.kappa = 2              # Concentration scaling factor for von Mises Fisher function.
        if directions is None:
            self.create_batch() # Create a batch of directions randomly
        else:
            self.directions = directions    # Use the directions provided to the function

        # Array to hold each input to the policy network over time
        #stimulus = np.zeros((self.total_dur, self.num_inputs, self.num_trials))
        stimulus = np.random.normal(par['input_mean'], par['noise_in'], size=(self.total_dur, self.num_inputs, self.num_trials))

        # 2) Set up the first input, which is the fixation input
        stimulus[0:self.total_dur-self.resp_dur,0,:] += self.tuning_height    # Fixation dot is on until the response period

        # 3) Calculate values for the motion tuned MT neurons
        # 3.1) Generate list of prefered directions (evenly distributed)
        pref_dirs = np.float64(np.arange(0,360,360/(par['num_motion_tuned'])))

        for tr in range(self.num_trials):    # For each trial
            for neu in range(par['num_motion_tuned']):  # For each neuron
                # Response of this neuron to sample stimulus for this trial
                d = np.cos(np.deg2rad(self.directions[tr, 0]-pref_dirs[neu]))   # Cos similarity between the two directions
                r = self.tuning_height*np.exp(self.kappa*d)/np.exp(self.kappa)                        # Tuning curve (vMF)
                stimulus[self.fix_dur:self.fix_dur+self.sample_dur,neu+1,tr] += r

                # Response of this neuron to test stimulus for this trial
                d = np.cos(np.deg2rad(self.directions[tr, 1]-pref_dirs[neu]))   # Cos similarity between the two directions
                r = self.tuning_height*np.exp(self.kappa*d)/np.exp(self.kappa)                        # Tuning curve (vMF)
                stimulus[self.fix_dur+self.sample_dur+self.delay_dur:self.fix_dur+self.sample_dur+self.delay_dur+self.test_dur,neu+1,tr] += r
        self.stimulus = np.float64(stimulus) # Convert to float64 from float32(default)

        # 4) Create the truth vector, which indicates for each trial whether or not the two direction are in the same category
        # Caclulate category of each stimulus for each trial
        cat = np.zeros(self.directions.shape)
        cat1_ind = np.logical_and(self.directions>self.boundary[0],self.directions<self.boundary[1])    # Indices for category 1 directions
        cat2_ind = np.logical_or(self.directions<self.boundary[0],self.directions>self.boundary[1])    # Indices for category 1 directions
        cat[cat1_ind] = 1
        cat[cat2_ind] = 2
        # Sanity check, each direction must have been asigned to a direction, no zero values should remain in this array
        if np.any(cat==0):
            raise TypeError('Some directions could not be assigned to any category')
        self.cat = cat
        # Create truth vetor, has a value of 1 if the two directions are a match, and a value of 2 if non-match
        truth = np.float64(np.diff(cat, axis=1)==0); truth[truth==0]=2
        self.truth = truth.squeeze() # Convert to float64 from float32(default)
        target = np.zeros((self.total_dur, 3, par['batch_train_size']))
        target[0:self.sample_dur+self.delay_dur+self.test_dur, 0, :] = 1    # Mark when we want the chosen action to be fixation
        target[self.sample_dur+self.delay_dur+self.test_dur:, 1, np.squeeze(truth==1)] = 1    # Mark when we want the chosen action to be match
        target[self.sample_dur+self.delay_dur+self.test_dur:, 2, np.squeeze(truth==2)] = 1    # Mark when we want the chosen action to be non_match
        self.target = target

    def run_trials(self, pol_x, val_x, stimulus, truth, extra_args):
        # Inputs:
        #   pol_x: Initial values for hidden units of the policy network
        #   val_x: Initial values for hidden units of the value network
        #   stimulus: The stimuli, # time poitns x 3 (fixation, rightward evidence, leftward evidence)
        #   truth: Vector that has (# trials) elements, each specifying true direction for that trial
        # extra_args: Extra arguments, depends on network type
        ############# Progress trial over time ##############
        # Unpack the extra arguments
        # Policy network
        if par['pol_unit_type'] == 'STSP':
            pol_syn_x_init = extra_args['pol_syn_x_init']
            pol_syn_u_init = extra_args['pol_syn_u_init']
        # Value network
        if par['val_unit_type'] == 'STSP':
            val_syn_x_init = extra_args['val_syn_x_init']
            val_syn_u_init = extra_args['val_syn_u_init']

        # Unstack input data across the time dimension
        input_data = tf.unstack(stimulus, axis=0)
        # Put truth in proper shape
        truth = tf.expand_dims(truth, axis=1)
        # Define tensor flow objects for fixation, so the selected action can be compared to
        fixate = tf.constant(np.zeros((par['batch_train_size'], 1)), dtype=np.float64)
        # Define match and non-match decisions, so action can be compared to
        match_choice = tf.constant(np.ones((par['batch_train_size'], 1)), dtype=np.float64) # actions=1 corresponds to match
        nonmatch_choice = tf.constant(2*np.ones((par['batch_train_size'], 1)), dtype=np.float64) # actions=2 corresponds to match
        # Array to keep history of hidden units activity (x=current, r=firing rate) of the policy network
        self.pol_x_history = []; self.pol_r_history = [];
        # Array to keep output of the policy network
        self.pol_out_history = []; self.pol_out_history0 = [];
        # Array to keep history of hidden units activity (x=current, r=firing rate) of the value Netwrok
        self.val_x_history = []; self.val_r_history = [];
        # Array to keep output of the value network
        self.val_out_history = []
        ############# Decision and reward array initiation##########
        # Array to hold the actions of the agent at any point in time
        # 0 is fixation, 1 is saccade right, 2 is saccade left
        # Array to keep all the selected actions
        self.actions = []
        self.action_array = []  # Array version of action (binary)
        # Array to keep track of the reward over time
        self.reward = []
        # Array to keep track of discount applied to each reward at each time point
        self.discount = []  # See page 17 of Song et al., 2017
        # Array to keep track of logpi, i.e. probability of the selected action at each point in time
        # This will be used to calculate J later
        self.logpi = []
        # Initialize a time mask to keep track of when each trial has been aborted or finished
        self.time_mask = []
        self.ideal = []
        # Array to reflect fixating
        temp_fixate = np.zeros((par['batch_train_size'], 3), dtype=np.bool_); temp_fixate[:, 0] = 1
        fixate_array = tf.constant(temp_fixate, dtype=tf.bool)
        # Array to reflect choosing match
        temp_match = np.zeros((par['batch_train_size'], 3), dtype=np.bool_); temp_match[:, 1] = 1
        match_array = tf.constant(temp_match, dtype=tf.bool)
        # Array to reflect choosing nonmatch
        temp_nonmatch = np.zeros((par['batch_train_size'], 3), dtype=np.bool_); temp_nonmatch[:, 2] = 1
        nonmatch_array = tf.constant(temp_nonmatch, dtype=tf.bool)

        self.temp1 = []; self.temp2 = []
        # Go over input at each point in time (this_u)
        self.t = 0   # Variable to keep track of time (in # itme points)
        cont_flag = tf.constant(np.ones((self.num_inputs, par['batch_train_size'])), dtype=tf.float64)   # Continuation of trial flag
        for this_u in input_data:
            '''
            1) Policy network:
            Given the input and previous hidden unit activity, get activity of hidden units at next time step
            x is the input current to each cell and r is be firing rate
            '''
            this_input = tf.multiply(cont_flag, this_u)
            if par['pol_unit_type'] == 'STSP':
                pol_x, pol_r, other_params = pol_cell(this_input, pol_x, pol_syn_u_init=pol_syn_u_init, pol_syn_x_init=pol_syn_x_init)
            else:
                pol_x, pol_r, other_params = pol_cell(this_input, pol_x)
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
            #pol_out = pol_out_0

            # Append current output of the policy network to its history
            self.pol_out_history.append(pol_out)
            self.pol_out_history0.append(pol_out_0)
            '''
            # 3) Choose action
            Given the output of the policy network, which specifies probabilities, choose an action
            '''
            # The multinomial will generate a number in [0, 2] range, 0=fixation, 1=match, 2=nonmatch
            # The multinomial function of tensorflow requires logits, hence the log
            this_action = tf.multinomial(tf.transpose(pol_out), 1) #tf.multinomial(tf.log(tf.transpose(pol_out)), 1)  # Do NOT remove the log!, or will produce samples not from the given distribution!
            this_action = tf.cast(this_action, dtype=tf.float64)
            # I just need to do it as an input to the value network, otherwise, having the actions vector as is for reward calculation is better
            self.actions.append(this_action)
            # 5) Given the selected action for each batch (trial), calculate the state of the system and its reward
            action_array = tf.constant(np.zeros((par['batch_train_size'], 3)), dtype=tf.bool)

            # Update the action array based on chosen actions
            temp1 = tf.logical_or(tf.logical_and(tf.tile(tf.equal(this_action, match_choice), [1, 3]), match_array), \
            tf.logical_and(tf.tile(tf.equal(this_action, nonmatch_choice), [1, 3]), nonmatch_array))
            temp2 = tf.logical_or(temp1, tf.logical_and(tf.tile(tf.equal(this_action, fixate), [1, 3]), fixate_array))
            action_array = tf.logical_or(action_array, temp2)
            self.temp1.append(temp1); self.temp2.append(temp2);
            action_array = tf.cast(action_array, dtype=tf.float64)
            # Update action in array form
            self.action_array.append(action_array)
            # Caclaulte reward
            if self.t<=(self.total_dur-self.resp_dur):   # During any period other than response period, action must be 0 (i.e. fixation)
                # If fixatiob is kept, reward is 0
                # If fixation is broken, reward is -1 (punishment)
                # So we just subtract 1 from the equality check
                this_reward = tf.cast(tf.equal(fixate, this_action), dtype=tf.float64) - 1

            elif self.t>(self.total_dur-self.resp_dur):    # During response period, action must be chosen as match or non-match to get a reward
                # If fixation is broken, reward is 0 (no punishment)
                # If made the correct decision, reward is 1
                # If made the wrong decision, reward is 0 (no punishment)
                #this_reward = tf.cast(tf.equal(truth, this_action), dtype=tf.float64) - 1 - 0*tf.cast(tf.equal(fixate, this_action), dtype=tf.float64)
                this_reward = tf.cast(tf.equal(truth, this_action), dtype=tf.float64) # - tf.cast(tf.not_equal(truth, this_action), dtype=tf.float64)
            else:
                print('Unknown time value!')
            # Should the trial continue? Update the cont_flag
            # As long as the obtained reward is 0, the trial continues
            cont_flag = tf.multiply(cont_flag, tf.tile(tf.cast(tf.equal(tf.transpose(this_reward), 0), dtype=tf.float64), [self.num_inputs, 1]))

            # Update time mask
            if self.t==0:    # Always include time point 0
                this_mask = tf.constant(np.ones((par['batch_train_size'], 1)), dtype=tf.bool)
                self.time_mask.append(this_mask)
                this_mask = tf.logical_and(this_mask, tf.equal(fixate, this_action))
            else:   # After time point 0
                    # Exclude a time point if it has already been excluded or ...
                    # if fixation gets broken
                self.time_mask.append(this_mask)
                this_mask = tf.logical_and(this_mask, tf.equal(fixate, this_action))

            # Append the current reward to the corresponding arrays
            self.reward.append(this_reward)

            # Calculate logpolicy (component in equation 3), i.e. P(selected action at each time point)
            #logpi = tf.multiply(pol_out, tf.cast(tf.transpose(action_array),dtype=tf.float64))
            logpi = tf.multiply(tf.nn.softmax(pol_out, 0), tf.cast(tf.transpose(action_array),dtype=tf.float64))
            logpi = tf.log(tf.reduce_sum(logpi, axis=0)+1e-7) #tf.log(tf.reduce_sum(logpi, axis=0))

            self.logpi.append(logpi)
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
            self.value_input = tf.concat([activity_input, action_input], axis=0)    # Concatenate the two along the zeroth axis
            # Calculate activity of hidden unit
            if par['val_unit_type'] == 'STSP':
                val_x, val_r, other_params = val_cell(self.value_input, val_x, val_syn_u_init=val_syn_u_init, val_syn_x_init=val_syn_x_init)
            else:
                val_x, val_r, other_params = val_cell(self.value_input, val_x)
            # Append current activity of the policy network units to their history
            self.val_x_history.append(tf.transpose(val_x))
            self.val_r_history.append(tf.transpose(val_r))

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
        self.logpi = tf.stack(self.logpi)


    def plot_trials(self):
        # Plot trials

        # Create T: Time vector in ms
        T = np.arange(0, self.total_dur*par['dt'], par['dt'])
        plt.plot(T, self.evid_right, 'b')
        plt.plot(T, self.evid_left, 'r')
        plt.plot(T, self.fix_cue, 'k')
        plt.legend(['Right', 'Left', 'Fixation'])
        plt.show()

    def perf(self):
        # This method calculates performance of the network when called using the
        # network weights to calculate performance of the agent on the RDM task

        # 1) Create a batch of trials with possible all directions to calculate performance for each direction as well as overall performance
        task.possible_dirs

        # % of break trials
        temp = np.sum(actions[:self.fix_dur+self.stim_dur, :], axis=0)
        # How many trials have a non-zero (i.e. not a fixation action) in the
        # period prior ot decision period divided by total number of trials
        self.breaks = np.sum(temp!=0)/actions.shape[1]

        # Reaction times
        # Find where first fixation was broken with reference to the go cue,
        # multiplied by time step
        RT = np.argmax(actions!=0, axis=0) - self.fix_dur - self.stim_dur
        self.RT = RT*par['dt']
        # All coherence values
        all_coh = np.unique(self.coh)
        # % Correct trials for each coherence value
        self.corr = np.zeros(all_coh.shape)
        # Wrong choice trials for each coherence value
        self.wrongs = np.zeros(all_coh.shape)
        for co in all_coh:
            # Find trials with this coherence value
            # Exclude trials where fixation is broken, or RT is Negative
            ind = np.where(np.logical_and(self.coh==co, RT>0)); ind= ind[0]
            these_trials = actions[:, ind]
            if these_trials.size == 0:
                self.corr[all_coh==co] = 0
                self.wrongs[all_coh==co] = 0
            else:
                #pdb.set_trace()
                # Find the first time point fixation is broken (actions is taken) for each coherence
                action_ind = np.argmax(these_trials!=0, axis=0)
                these_actions = these_trials[action_ind, :]
                self.corr[all_coh==co] = np.sum(these_actions==self.truth[ind])/np.sum(self.coh==co)
                self.wrongs[all_coh==co] = np.sum(these_actions!=self.truth[ind])/np.sum(self.coh==co)

    def update_params(self):
        # Specify task and network parameters based on task needs
        par['dead_time'] = 0
        par['fix_dur'] = 50         # Fixaion priod duration in ms
        par['fix_time'] = par['fix_dur']
        par['sample_dur'] = 200     # Sample stimulus period duration in ms
        par['sample_time'] = par['sample_dur']
        par['delay_dur'] = 100      # Delay priod duration in ms
        par['delay_time'] = par['delay_dur']
        par['test_dur'] = 200       # Test period duration in ms
        par['resp_dur'] = 400       # Response period duration in ms
        par['mask_duration'] = par['test_dur']
        par['num_time_steps'] = np.int32((par['fix_dur'] + par['sample_dur'] + par['delay_dur'] + par['test_dur'] + par['resp_dur'])/par['dt'])


        # Policy network shape
        par['num_motion_tuned'] = 36                          # Number of MT neurons
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
        # DCM only has three possible actions: Fixate, match, non-match

        # Initialize input weights for the policy network
        '''
        par['pol_w_in0'] = initialize([par['pol_n_hidden'], par['pol_num_input']], par['connection_prob'])
        # Initialize recurrent weights for the policy network
        par['pol_w_rnn0'] = np.eye(par['pol_n_hidden'], dtype=np.float64)
        par['pol_b_rnn0'] = np.zeros((par['pol_n_hidden'], 1), dtype=np.float64)
        # Initialize output weights for the policy network
        par['pol_w_out0'] =initialize([par['pol_n_output'], par['pol_n_hidden']], par['connection_prob'])
        par['pol_b_out0'] = np.zeros((par['pol_n_output'], 1), dtype=np.float64)
        '''
        c = 0.1
        par['pol_w_in0'] = np.float64(np.random.uniform(-c,c, size = [par['pol_n_hidden'], par['pol_num_input']]))
        par['pol_w_rnn0'] = np.float64(np.random.uniform(-c,c, size = [par['pol_n_hidden'], par['pol_n_hidden']]))
        par['pol_b_rnn0'] = np.float64(np.random.uniform(-c,c, size = [par['pol_n_hidden'], 1]))
        par['pol_w_out0'] = np.float64(np.random.uniform(-c,c, size = [par['pol_n_output'], par['pol_n_hidden']]))
        par['pol_b_out0'] = np.zeros([par['pol_n_output'], 1], dtype = np.float64)
        # Value network shape
        # Reminder: Value network only has one output: expected reward (Song et al., 2017)
        # Value network gets as input activity of all the hidden units in the policy network
        # as well as agent (i.e. which decisions have zero values (not made) and which one has a value of one (made))
        par['val_num_input'] = par['pol_n_hidden'] + par['pol_n_output']   # Number of inputs to the value network
        par['val_n_hidden'] = 100  # Number of hidden units in the recurrent network
        par['val_n_output'] = 1    # Number of outputs for the value network

        # Initialize input weights for the value network
        '''
        par['val_w_in0'] = initialize([par['val_n_hidden'], par['val_num_input']], par['connection_prob'])
        # Initialize recurrent weights for the value network
        par['val_w_rnn0'] = np.eye(par['val_n_hidden'], dtype=np.float64)
        par['val_b_rnn0'] = np.zeros((par['val_n_hidden'], 1), dtype=np.float64)
        # Initialize output weights for the value network
        par['val_w_out0'] =initialize([par['val_n_output'], par['val_n_hidden']], par['connection_prob'])
        par['val_b_out0'] = np.zeros((par['val_n_output'], 1), dtype=np.float64)
        '''
        par['val_w_in0'] = np.float64(np.random.uniform(-c,c, size = [par['val_n_hidden'], par['val_num_input']]))
        par['val_w_rnn0'] = np.float64(np.random.uniform(-c,c, size = [par['val_n_hidden'], par['val_n_hidden']]))
        par['val_b_rnn0'] = np.float64(np.random.uniform(-c,c, size = [par['val_n_hidden'], 1]))
        par['val_w_out0'] = np.float64(np.random.uniform(-c,c, size = [par['val_n_output'], par['val_n_hidden']]))
        par['val_b_out0'] = np.float64(np.random.uniform(-c,c, size = [1, 1]))

        # Added by Pantea
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
