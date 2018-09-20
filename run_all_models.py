import numpy as np
from parameters import *
import model
import sys
import pdb

task_list = ['DMC']

pdb.set_trace()
for task in task_list:
    for j in range(25):
        print('Training network on ', task,' task, network model number ', j)
        save_fn = task + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn}
        update_parameters(updates)
        # Keep the try-except clauses to ensure proper GPU memory release
        try:
            """
            # GPU designated by first argument (must be integer 0-3)
            try:
                print('Selecting GPU ',  sys.argv[1])
                assert(int(sys.argv[1]) in [0,1,2,3])
            except AssertionError:
                quit('Error: Select a valid GPU number.')
            """
            pdb.set_trace()
            # Run model
            #model.train_and_analyze(sys.argv[1])
            #Commented by Pantea to get rid of gpu stuff
            model.train_and_analyze()
        except KeyboardInterrupt:
            quit('Quit by KeyboardInterrupt')


# Command for observing python processes:
# ps -A | grep python
