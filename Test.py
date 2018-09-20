'''
Test.py
Whenever I need to test something, I will use this script,
not used by the actual program
'''

import numpy as np
import tensorflow as tf
import pdb

tf.reset_default_graph()

# Tensorflow finds the supported CPU and GPU devices you can use
config = tf.ConfigProto()


# See if softmax matters
p = tf.constant([[1, 2, 1]], dtype=tf.float64)

# Without softmax
out1 = tf.multinomial(p, 1)

# With softmax
p_log = tf.log(p)
out2 = tf.multinomial(p_log, 1)

with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    o1 = []
    o2 = []
    for it in range(10000):

        temp1, temp2 = sess.run([out1, out2])
        o1.append(temp1)
        o2.append(temp2)
o1 = np.squeeze(np.array(o1))
o2 = np.squeeze(np.array(o2))
pdb.set_trace()
