"""
This piece of code is written to test the polynomial function of the tensorflow and compare it
to python's sampler
"""

import numpy as np
import tensorflow as tf
import pdb


tf.reset_default_graph()
N = 10000   # Number of samples
Prob = np.array([[-3.0, -3.0, 1.0]])
Prob_softmax = tf.log(tf.nn.softmax(Prob))
#P = tf.constant(Prob_softmax, dtype=tf.float32) # Probablity distribution
sampler = tf.multinomial(Prob_softmax, 10000)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    samples, real_P = sess.run([sampler, Prob_softmax])


Pout = np.zeros((1, 3))   # Probability distribution of outputs
Pout[0, 0] = np.sum(samples==0)/N
Pout[0, 1] = np.sum(samples==1)/N
Pout[0, 2] = np.sum(samples==2)/N
print(Prob)
print(real_P)
print(Pout)
