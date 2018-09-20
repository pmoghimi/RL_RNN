# This script is for whatever I am currently working on
# Do not store permanent code here

import pickle
import matplotlib.pyplot as plt

with open('.\DMC\DMC0.pkl', 'rb') as f:
    D = pickle.load(f)

D.keys()
plt.plot(D['neuronal_pref_dir'][0,0,:]);
plt.show()
