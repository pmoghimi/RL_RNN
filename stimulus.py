import numpy as np
import matplotlib.pyplot as plt
from parameters import *


class Stimulus:

    def __init__(self):

        # generate tuning functions
        self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()


    def generate_trial(self, test_mode = False):
        trial_info = self.generate_basic_trial(test_mode)
        return trial_info

    def generate_basic_trial(self, test_mode):

        """
        Generate a delayed matching task
        Goal is to determine whether the sample stimulus, possibly manipulated by a rule, is
        identicical to a test stimulus
        Sample and test stimuli are separated by a delay
        """
        # range of variable delay, in time steps
        var_delay_max = par['variable_delay_max']//par['dt']

        # rule signal can appear at the end of delay1_time
        trial_length = par['num_time_steps']

        # end of trial epochs
        eodead = par['dead_time']//par['dt']
        eof = (par['dead_time']+par['fix_time'])//par['dt']
        eos = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        eod = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

        # end of neuron indices
        emt = par['num_motion_tuned']
        eft = par['num_fix_tuned']+par['num_motion_tuned']
        ert = par['num_fix_tuned']+par['num_motion_tuned'] + par['num_rule_tuned']

        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']

        trial_info = {'desired_output'  :  np.zeros((par['n_output'], trial_length, par['batch_train_size']),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, par['batch_train_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'rule'            :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'catch'           :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'probe'           :  np.zeros((par['batch_train_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['n_input'], trial_length, par['batch_train_size']))}


        # set to mask equal to zero during the dead time
        trial_info['train_mask'][:eodead, :] = 0

        # If the DMS and DMS rotate are being performed together,
        # or if I need to make the test more challenging, this will eliminate easry test directions
        # If so, reduce set of test stimuli so that a single strategy can't be used
        #limit_test_directions = par['trial_type']=='DMS+DMRS'

        for t in range(par['batch_train_size']):

            """
            Generate trial paramaters
            """
            sample_dir = np.random.randint(par['num_motion_dirs'])
            if test_mode:
                test_dir = np.random.randint(par['num_motion_dirs'])
            rule = np.random.randint(par['num_rules'])
            if par['trial_type'] == 'DMC' or (par['trial_type'] == 'DMS+DMC' and rule == 1) or (par['trial_type'] == 'DMS+DMRS+DMC' and rule == 2):
                # for DMS+DMC trial type, rule 0 will be DMS, and rule 1 will be DMC
                current_trial_DMC = True
            else:
                current_trial_DMC = False

            match = np.random.randint(2)
            catch = np.random.rand() < par['catch_trial_pct']

            """
            Generate trial paramaters, which can vary given the rule
            """
            if par['num_rules'] == 1:
                match_rotation = int(par['num_motion_dirs']*par['rotation_match']/360)
            else:
                match_rotation = int(par['num_motion_dirs']*par['rotation_match'][rule]/360)

            """
            Determine the delay time for this trial
            The total trial length is kept constant, so a shorter delay implies a longer test stimulus
            """
            eod_current = eod

            # set mask to zero during transition from delay to test
            trial_info['train_mask'][eod_current:eod_current+mask_duration, t] = 0

            """
            Generate the sample and test stimuli based on the rule
            """
            # DMC
            # categorize between two equal size, contiguous zones
            sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
            if match == 1: # match trial
                # do not use sample_dir as a match test stimulus
                dir0 = int(sample_cat*par['num_motion_dirs']//2)
                dir1 = int(par['num_motion_dirs']//2 + sample_cat*par['num_motion_dirs']//2)
                #possible_dirs = np.setdiff1d(list(range(dir0, dir1)), sample_dir)
                possible_dirs = list(range(dir0, dir1))
                test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
            else:
                test_dir = sample_cat*(par['num_motion_dirs']//2) + np.random.randint(par['num_motion_dirs']//2)
                test_dir = np.int_((test_dir+par['num_motion_dirs']//2)%par['num_motion_dirs'])
            """
            Calculate neural input based on sample, tests, fixation, rule, and probe
            """
            # SAMPLE stimulus
            trial_info['neural_input'][:emt, eof:eos, t] += np.reshape(self.motion_tuning[:,sample_dir],(-1,1))

            # TEST stimulus
            trial_info['neural_input'][:emt, eod_current:, t] += np.reshape(self.motion_tuning[:,test_dir],(-1,1))
            #pdb.set_trace()
            """
            Determine the desired network output response
            """
            trial_info['desired_output'][0, eodead:eod_current, t] = 1
            trial_info['train_mask'][ eod_current:, t] = 1 # can use a greater weight for test period if needed
            if match == 0:
                trial_info['desired_output'][1, eod_current:, t] = 1
            else:
                trial_info['desired_output'][2, eod_current:, t] = 1

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

        print('num_receptive_fields', par['num_receptive_fields'])

        """
        Generate tuning functions for the Postle task
        """
        motion_tuning = np.zeros((par['num_motion_tuned'], par['num_receptive_fields'], par['num_motion_dirs']))
        fix_tuning = np.zeros((par['num_fix_tuned'], par['num_receptive_fields']))
        rule_tuning = np.zeros((par['num_rule_tuned'], par['num_rules']))

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

        for n in range(par['num_fix_tuned']):
            for i in range(2):
                if n%2 == i:
                    fix_tuning[n,i] = par['tuning_height']

        for n in range(par['num_rule_tuned']):
            for i in range(par['num_rules']):
                if n%par['num_rules'] == i:
                    rule_tuning[n,i] = par['tuning_height']


        return np.squeeze(motion_tuning), fix_tuning, rule_tuning


    def plot_neural_input(self, trial_info):

        print(trial_info['desired_output'][ :, 0, :].T)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(0,400+500+2000,par['dt'])
        t -= 900
        t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
        #im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
        im = ax.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='none')
        #plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
        ax.set_yticks([0, 9, 18, 27])
        ax.set_yticklabels([0,90,180,270])
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Motion direction')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Motion input')
        plt.show()
        plt.savefig('stimulus.pdf', format='pdf')
