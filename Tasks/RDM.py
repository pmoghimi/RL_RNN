'''
Random Dot Movement task
Refs: Kiani et al., 2008 and Song et al., 2017
'''
import numpy as np
from parameters import *
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
from TaskTools import *

class rdm:
    def __init__(self, num_trials):
        '''
        Initializes the RDM taks object
        Task is divided to the following epochs:
        1) Fixation period:
            fixation dot on, agent has to Fixate
            duration: 750ms (from Song et al., 2017)
        2) Stimulus period:
            fixation dot on, stimulus on, agent has to Fixate
            duration: 800ms,
            Song et al., 2017 used arange of values, but for now let's fix it at 800ms
        3) Decision period:
            fixation dot off, the agent has to indicate its decision
            duration: 500ms (Song et al., 2017)

        Inputs:
        coh: Coherence of the dot movements
            Specifies percentage of dots moving to the RIGHT
            Positive: Movement to the RIGHT
            Negative: Movement to the LEFT
        '''
        """
        Reset TensorFlow before running anything
        """
        #tf.reset_default_graph()
        # Set parameters for the RDM task
        ### Stimulus ###

        self.num_trials = num_trials # Number of trials (train batch size)
        # Update the par variable that contains parameters according to task needs
        self.update_params()
        # With the new parameters update the depndencies (function in parameters.py)
        update_dependencies()

        # Convert time parameters from ms to number of steps
        self.fix_dur = par['fix_dur'] // par['dt'] # Fixaion priod duration in number of steps
        self.stim_dur = par['stim_dur'] // par['dt'] # Stimulus period duration in number of steps
        self.dec_dur = par['dec_dur'] // par['dt'] # Decision priod duration in number of steps
        # Total trial duration in number of steps
        self.total_dur = self.fix_dur + self.stim_dur + self.dec_dur
        # Numer of stimulus inputs for this task: fixation, right ward motion, left ward motion
        self.num_inputs = 3

    def create_batch(self):
        """
        This function creates a batch of trials
        It will randomly sample coherence values from a unifotm distribution between 1 an -1
        Number of coherence values chosen will be equal to number of iterations
        """
        # Acceptable coherence values
        # Negative: left ward movement
        # Positive: Right ward movement
        Acceptable_coh_values = np.linspace(-1, 1, 6)
        # Sample from these values,
        # number of samples is equal to number of trials par['train_batch_size']
        rand_ind = np.random.randint(0, np.size(Acceptable_coh_values), self.num_trials)
        coh = Acceptable_coh_values[rand_ind]
        return coh


    def create_stimulus(self):
        """
        This function creates a batch of trials everytime it is called
        Output:
            stimulus, a tensor, size = #time points x 3 x #trials
                stimulus[:,0,:] is the fixation cue, one when fixation requried, zero otherwise
                stimulus[:,1,:] is evidence for rightward motion over time
                stimulus[:,2,:] is evidence for left ward motion over time
        """
        # Calculate true direction based on coherence
        # Coh>0, true direction is 1 or rightward
        # Coh<0, true direction is -1 or leftward
        # Coh=0, true direction is picked randomly
        # Create a bunch of coherence values randomly
        coh = self.create_batch()
        self.coh = coh
        # Map + coh values to 1 and - coh values to -1
        truth = 2 * (coh>0) - 1
        # Choose a random true direction for - values
        truth[coh==0] = np.random.randint(1, 3) # Generate a random number that is either 1 or 2
        truth[coh==0] = -2*truth[coh==0] + 3    # If the generated number is 1, the true direction is 1, if the random number is 2, true direction is -1
        self.truth = np.float64(truth)  # Covert to float64 from float32 (default)
        '''
        if coh>0:      # Right
            self.true_dir = 1
        elif coh<0:    # Left
            self.true_dir = -1
        else:               # Picked randomly at 0 coh
            self.true_dir = np.random.randint(1, 3) # Generate a random number that is either 1 or 2
            self.true_dir = -2*self.true_dir + 3    # If the generated number is 1, the true direction is 1, if the random number is 2, true direction is -1
        '''
        # Array to hold evidence (percentage of dots moving to the right) over times
        evid_right = np.zeros((self.total_dur, 1, self.num_trials))
        # During stimulation period, set evidence for right, with:
        # mean of (1+coh)/2, i.e. at zero coherence 0.5 of the dots are oving RIGHT
        # sigma equal tp sqrt(2*100*0.01)/dt
        evid_right[self.fix_dur+1:self.fix_dur+self.stim_dur+1, 0, :] = np.random.normal((1+coh)/2, scale=par['sigma']/par['dt'], size=(self.stim_dur, self.num_trials))

        # Array to hold evidence (percentage of dots moving to the left) over times
        evid_left = np.zeros((self.total_dur, 1, self.num_trials))
        # During stimulation period, set evidence for left, with:
        # mean of (1-coh)/2, i.e. at zero coherence 0.5 of the dots are oving LEFT
        # sigma equal tp sqrt(2*100*0.01)/dt
        evid_left[self.fix_dur+1:self.fix_dur+self.stim_dur+1, 0, :] = np.random.normal((1-coh)/2, scale=par['sigma']/par['dt'], size=(self.stim_dur, self.num_trials))

        # Array to hold fixation cue input
        # Fixation cue will be on (equal to 1) during fixation and stimulus periods
        fix_cue = np.zeros((self.total_dur, 1, self.num_trials))
        fix_cue[0:self.fix_dur+self.stim_dur+1, 0, :] = 1
        # Visual stimulus
        stimulus = np.concatenate((fix_cue, evid_right, evid_left), axis = 1)
        self.stimulus = np.float64(stimulus)    # Convert to float64 from float32(default)
        # Create a tensor flow object for the input by concatenating the three inputs
        #self.stimulus = np.concatenate((fix_cue, evid_right, evid_left), axis = 1)
        # self.stimulus = tf.constant(all_inputs, dtype=np.float32)

        """
        TEMPORARY: Caculate a target output just in case, for supervised learning testing
        """
        # Create the desired output
        target = np.zeros((self.total_dur, 3, par['batch_train_size']))# + np.log(0.025)
        # Make the desired action fixation during fixation and stim periods
        target[0:(self.fix_dur+self.stim_dur), 1, :] = 1 #np.log(0.95)
        # Turn right
        target[(self.fix_dur+self.stim_dur)+1:, 2, coh>=0] = 1 #np.log(0.95)
        # Turn left
        target[(self.fix_dur+self.stim_dur)+1:, 0, coh<0] = 1 #np.log(0.95)
        self.target = np.float64(target)    # Convert to float64 form float32 (default)


    def run_trials(self, pol_x, val_x, stimulus, truth):
        # Inputs:
        #   pol_x: Initial values for hidden units of the policy network
        #   val_x: Initial values for hidden units of the value network
        #   stimulus: The stimuli, # time poitns x 3 (fixation, rightward evidence, leftward evidence)
        #   truth: Vector that has (# trials) elements, each specifying true direction for that trial
        ############# Progress trial over time ##############
        # Unstack input data across the time dimension
        input_data = tf.unstack(stimulus, axis=0)
        # Put truth in proper shape
        truth = tf.expand_dims(truth, axis=1)
        # Define tensor flow objects for fixation, so the selected action can be compared to
        fixate = tf.constant(np.zeros((par['batch_train_size'], 1)), dtype=np.float64)
        # Define left and right direction choices, so action can be compared to
        right_choice = tf.constant(np.ones((par['batch_train_size'], 1)), dtype=np.float64)
        left_choice = tf.constant(-np.ones((par['batch_train_size'], 1)), dtype=np.float64)
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
        self.time_mask = [] #tf.constant(np.ones((self.total_dur, par['batch_train_size'])), dtype=np.int32)
        self.ideal = []
        # Initialize an array for the input to the value network that reflects the chosen action
        # The array keeps track of the chosen action at each time point
        # 3 actions are possible, hence number 3 at the third dimension
        # A [1 0 0] at the third dimension: fixate
        # A [0 1 0] at the third dimension: Go right
        # A [0 0 1] at the third dimension: Go left
        # self.action_array = tf.constant(np.zeros((self.total_dur, par['batch_train_size'], 3)), dtype=np.int32)
        # self.action_array = tf.constant(np.zeros((par['batch_train_size'], 3)), dtype=np.float32)
        temp_fixate = np.zeros((par['batch_train_size'], 3), dtype=np.bool_); temp_fixate[:, 1] = 1
        fixate_array = tf.constant(temp_fixate, dtype=tf.bool)
        # Array to reflect choosing the right direction
        #temp_right = np.zeros((self.total_dur, par['batch_train_size'], 3))
        #temp_right[self.t, self.b, 1] = 1
        temp_right = np.zeros((par['batch_train_size'], 3), dtype=np.bool_); temp_right[:, 2] = 1
        right_array = tf.constant(temp_right, dtype=tf.bool)
        # Array to reflect choosing the left direction
        #temp_left = np.zeros((self.total_dur, par['batch_train_size'], 3))
        #temp_left[self.t, self.b, 2] = 1
        temp_left = np.zeros((par['batch_train_size'], 3), dtype=np.bool_); temp_left[:, 0] = 1
        left_array = tf.constant(temp_left, dtype=tf.bool)

        self.temp_l = []; self.temp_r = []; self.temp1 = []; self.temp2 = []
        # Go over input at each point in time (this_u)
        self.t = 0   # Variable to keep track of time (in # itme points)
        cont_flag = tf.constant(np.ones((3, par['batch_train_size'])), dtype=tf.float64)   # Continuation of trial flag
        for this_u in input_data:
            '''
            1) Policy network:
            Given the input and previous hidden unit activity, get activity of hidden units at next time step
            x is the input current to each cell and r is be firing rate
            '''
            #pdb.set_trace()
            this_input = tf.multiply(cont_flag, this_u)
            pol_x, pol_r, other_params = pol_cell(this_input, pol_x)
            # Append current activity of the policy network units to their history
            self.pol_x_history.append(tf.transpose(pol_x))
            self.pol_r_history.append(tf.transpose(pol_r))
            '''
            # 2) Policy network:
            Given the hidden state firing rate at time t, get output at time t (policy)
            '''
            with tf.variable_scope('policy_output', reuse=True):
                pol_W_out = tf.get_variable('pol_W_out', dtype=tf.float64)
                pol_b_out = tf.get_variable('pol_b_out', dtype=tf.float64)
            pol_out_0 = tf.matmul(pol_W_out, pol_r) + pol_b_out   # Linear part, equation 6
            pol_out = tf.nn.softmax(pol_out_0, 0)  # Softmax part, equation 7
            #pol_out = pol_out_0
            ############ Create ideal pol out to see performance of the system for checking purposes
            bi=1e-20
            ideal_pol_out = np.zeros((3, par['batch_train_size'])) + bi
            if self.t<=(self.stim_dur+self.fix_dur):   # During fixation period, action must be 0 (i.e. fixation)
                ideal_pol_out[1,:] = 1-2*bi
                ideal = tf.constant(ideal_pol_out, dtype=np.float64)
            if self.t>(self.stim_dur+self.fix_dur):    # During decision period, action must be making a saccade to thr right direction to get a reward
                #pdb.set_trace()
                temp_l = tf.equal(truth, tf.constant(-np.ones((par['batch_train_size'], 1)), dtype=tf.float64))
                temp_l = tf.transpose(temp_l)
                temp_l0 = np.zeros_like(ideal_pol_out); temp_l0[0,:] = 1;
                temp_l = tf.logical_and(tf.tile(temp_l, [3, 1]), tf.constant(temp_l0, dtype=tf.bool))
                temp_r = tf.equal(truth, tf.constant(np.ones((par['batch_train_size'], 1)), dtype=tf.float64))
                temp_r = tf.transpose(temp_r)
                temp_r0 = np.zeros_like(ideal_pol_out); temp_r0[2,:] = 1;
                temp_r = tf.logical_and(tf.tile(temp_r, [3, 1]), tf.constant(temp_r0, dtype=tf.bool))
                ideal = tf.constant(ideal_pol_out, dtype=np.float64) + (1-3*bi)*tf.cast(tf.logical_or(temp_l, temp_r), dtype=tf.float64)
                self.temp_l.append(temp_l)
                self.temp_r.append(temp_r)
            #pdb.set_trace()
            pol_out = 1*pol_out + 0*ideal

            self.ideal.append(ideal)

            # Append current output of the policy network to its history
            self.pol_out_history.append(pol_out)
            self.pol_out_history0.append(pol_out_0)
            '''
            # 3) Choose action
            Given the output of the policy network, which specifies probabilities, choose an action
            '''

            # The multinomial will generate a number in [0, 2] range, by subtracting it by 1, we bring it to the [-1, 1] range
            # The multinomial function of tensorflow requires logits, hence the log
            this_action = tf.multinomial(tf.log(tf.transpose(pol_out)), 1) - 1  # Do not remove the log!, or will produce samples not from the given distribution!
            this_action = tf.cast(this_action, dtype=tf.float64)
            # I just need to do it as an input to the value network, otherwise, having the actions vector as is for reward calculation is better
            self.actions.append(this_action)
            # 5) Given the selected action for each batch (trial), calculate the state of the system and its reward
            action_array = tf.constant(np.zeros((par['batch_train_size'], 3)), dtype=tf.bool)

            # Update the action array based on chosen actions
            temp1 = tf.logical_or(tf.logical_and(tf.tile(tf.equal(this_action, right_choice), [1, 3]), right_array), \
            tf.logical_and(tf.tile(tf.equal(this_action, left_choice), [1, 3]), left_array))
            temp2 = tf.logical_or(temp1, tf.logical_and(tf.tile(tf.equal(this_action, fixate), [1, 3]), fixate_array))
            action_array = tf.logical_or(action_array, temp2)
            self.temp1.append(temp1); self.temp2.append(temp2);
            action_array = tf.cast(action_array, dtype=tf.float64)
            # Update action in array form
            self.action_array.append(action_array)
            # Caclaulte reward
            if self.t<=(self.stim_dur+self.fix_dur):   # During fixation period, action must be 0 (i.e. fixation)
                # If fixatiob is kept, reward is 0
                # If fixation is broken, reward is -1 (punishment)
                # So we just subtract 1 from the equality check
                this_reward = tf.cast(tf.equal(fixate, this_action), dtype=tf.float64) - 1

            if self.t>(self.stim_dur+self.fix_dur):    # During decision period, action must be making a saccade to thr right direction to get a reward
                # If fixation is broken, reward is 0 (no punishment)
                # If saccade to the correct target, reward is 1
                # If saccade to the wrong target, reward is 0 (no punishment)
                this_reward = 1*tf.cast(tf.equal(truth, this_action), dtype=tf.float64) #- 1 #*tf.cast(tf.equal(fixate, this_action), dtype=tf.float64)

            # Should the trial continue? Update the cont_flag
            # As long as the obtained reward is 0, the trial continues
            cont_flag = tf.multiply(cont_flag, tf.tile(tf.cast(tf.equal(tf.transpose(this_reward), 0), dtype=tf.float64), [3, 1]))

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
            logpi = tf.multiply(pol_out, tf.cast(tf.transpose(action_array),dtype=tf.float64))
            logpi = tf.log(tf.reduce_sum(logpi, axis=0))

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
            val_x, val_r, other_params = val_cell(self.value_input, val_x)
            # Append current activity of the policy network units to their history
            self.val_x_history.append(tf.transpose(val_x))
            self.val_r_history.append(tf.transpose(val_r))

            '''
            5) Value network:
            Given the hidden state activity at time t, get output at time t (predicted reward)
            '''

            with tf.variable_scope('value_output', reuse=True):
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

    def perf(self, actions):
        # This method calculates performance of the network when called using the
        # network weights to calculate performance of the agent on the RDM task
        # Inputs:
        #   actions: actions chosen at each point in time for each trial
        # Outputs:

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
        par['sigma'] = 0 #np.sqrt(2*100*0.01**2) # Taken from rmd_fixed from the code of Song et al., 2017
        # Reason: Apparently it might give people an ulcer to specify that in their paper!
        par['fix_dur'] = 20 #750 # Fixaion priod duration in ms
        par['stim_dur'] = 100 #800 # Stimulus period duration in ms
        par['dec_dur'] = 500 #500 # Decision priod duration in ms


        # Policy network shape
        par['pol_num_input'] = 3    # Number of inputs to the policy network
        # Inputs fo rhte RDM task are: Fixation cue, right evidence, left evidence
        par['pol_n_hidden'] = 100   # Number of hidden units in the recurrent network
        par['pol_n_output'] = 3     # Number of outputs for the policy network
        par['n_output'] = par['pol_n_output']
        # RDM only has three possible actions: Fixate, choose right, choose left
        # Initialize input weights for the policy network
        par['pol_w_in0'] = initialize([par['pol_n_hidden'], par['pol_num_input']], par['connection_prob'])
        # Initialize recurrent weights for the policy network


        # Sing et al. 2017 way of initializing recurrent weights
        K   = int(par['connection_prob']*par['pol_n_hidden'])
        idx = np.arange(par['pol_n_hidden'])
        # Wrec
        M = np.zeros((par['pol_n_hidden'], par['pol_n_hidden']), dtype=np.float64)
        for j in range(M.shape[1]):
            M[np.random.permutation(idx)[:K],j] = 1
        par['pol_w_rnn0'] = M

        par['pol_w_rnn0'] = np.eye(par['pol_n_hidden'], dtype=np.float64)

        par['pol_b_rnn0'] = np.zeros((par['pol_n_hidden'], 1), dtype=np.float64)
        # Initialize output weights for the policy network
        par['pol_w_out0'] =initialize([par['pol_n_output'], par['pol_n_hidden']], par['connection_prob'])
        par['pol_b_out0'] = np.zeros((par['pol_n_output'], 1), dtype=np.float64)


        # Value network shape
        # Reminder: Value network only has one output: expected reward (Song et al., 2017)
        # Value network gets as input activity of all the hidden units inthe policy network
        # as well as agent (i.e. which decisions have zero values (not made) and which one has a value of one (made))
        par['val_num_input'] = par['pol_n_hidden'] + par['pol_n_output']   # Number of inputs to the value network
        par['val_n_hidden'] = 100  # Number of hidden units in the recurrent network
        par['val_n_output'] = 1    # Number of outputs for the value network
        # Initialize input weights for the value network
        par['val_w_in0'] = initialize([par['val_n_hidden'], par['val_num_input']], par['connection_prob'])
        # Initialize recurrent weights for the value network

        K   = int(1*par['val_n_hidden'])
        idx = np.arange(par['val_n_hidden'])
        # Wrec
        M = np.zeros((par['val_n_hidden'], par['val_n_hidden']), dtype=np.float64)
        for j in range(M.shape[1]):
            M[np.random.permutation(idx)[:K],j] = 1
        par['val_w_rnn0'] = M

        #par['val_w_rnn0'] = np.eye(par['val_n_hidden'], dtype=np.float64)
        par['val_b_rnn0'] = np.zeros((par['val_n_hidden'], 1), dtype=np.float64)
        # Initialize output weights for the value network
        par['val_w_out0'] =initialize([par['val_n_output'], par['val_n_hidden']], par['connection_prob'])
        par['val_b_out0'] = np.zeros((par['val_n_output'], 1), dtype=np.float64)
