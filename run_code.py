# run_code
import Tasks.DMC
from Tasks.DMC import dmc
import matplotlib.pyplot as plt
import numpy as np
import pdb
#import model_works
import model_debug
#from model import Model
#import model_RL
from parameters import *
import os
import sys

"""
Reset TensorFlow before running anything
"""
tf.reset_default_graph()
par['trial_type'] = 'DMC'
A = dmc(par['batch_train_size'])
task_list = ['DMC']
# model.main(A)

#Check to see if a gpu id is provided, otherwise, assign a None value to it
if len(sys.argv)==1: # No gpu id provided
    gpu_id = None
elif len(sys.argv)==2:    # If a gpu id is provided
    gpu_id = sys.argv[1]

for j in range(0,10):
    for task in task_list:
        print('Training network on ', task,' task, network model number ', j)
        # Path for saving results
        par['save_path'] = task+'/'+str(j)+'/'
        # Create a folder to save results if it does not already exist
        if not os.path.exists(par['save_path']):
            os.makedirs(par['save_path'])
        #save_fn = task + '_' + str(j) + '.pkl'
        save_fn = 'Results.pkl'
        par['trial_type'] = task
        par['save_fn'] = save_fn
        #pdb.set_trace()
        if not os.path.isfile(par['save_path'] + par['save_fn']):
            model_debug.main(A, gpu_id)
        else:
            print('Network model number ', j ,' already exists, skipping...')

'''
TO DO LIST:
1) Implement biological constrains on RL, i.e. Dale's law, read out from exitatory weights only, etc.
2) Implement short term plasticity, compare performance with and without short term plasticity
3) Compare RL with supervised per Nick and supervised per Wang lab (r and x separated)
4) Play with learning rates with the supervised method
5) Introduce RFs, spatial aspects, potentially Clunen stuff
6) Implement linearization and state space representation and dPCA
7) Introduce weight sparsity (i.e. add L2 norm of the recurrent weights to the cost function) for the policy network
8) Explore order of training for policy and baseline networks, Wang did baseline first, why? does it matter?
'''
