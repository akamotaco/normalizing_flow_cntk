# https://github.com/abdulfatir/normalizing-flows/blob/master/notebooks/PlanarFlow-Example1.ipynb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tfd = tf.contrib.distributions
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# tf.set_random_seed(42)

K = 16
eps = 1e-7

def true_density(z):
    z1, z2 = z[:, 0], z[:, 1]
    norm = tf.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = tf.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = tf.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - tf.log(exp1 + exp2)
    return tf.exp(-u)

# m = lambda x: -1 + tf.log(1 + tf.exp(x))
h = lambda x: tf.tanh(x)
h_prime = lambda x: 1 - tf.tanh(x) ** 2
base_dist = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
z_0 = base_dist.sample(500)
z_prev = z_0
sum_log_det_jacob = 0.

##########################################################
# for i in range(K):
#     with tf.variable_scope('layer_%d' % i):
#         u = tf.get_variable('u', dtype=tf.float32, shape=(1, 2))
#         w = tf.get_variable('w', dtype=tf.float32, shape=(1, 2))
#         b = tf.get_variable('b', dtype=tf.float32, shape=())
#         u_hat = (m(tf.tensordot(w, u, 2)) - tf.tensordot(w, u, 2)) * (w / tf.norm(w)) + u
#         affine = h_prime(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b) * w
#         sum_log_det_jacob += tf.log(eps + tf.abs(1 + tf.reduce_sum(affine * u_hat, -1)))
#         z_prev = z_prev + u_hat * h(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b)
##############################################################
for i in range(K):
    with tf.variable_scope('layer_%d' % i):
        u = tf.get_variable('u', dtype=tf.float32, shape=(1, 2))
        w = tf.get_variable('w', dtype=tf.float32, shape=(1, 2))
        b = tf.get_variable('b', dtype=tf.float32, shape=())
        psi = h_prime(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b) * w
        det_jacob = tf.abs(1 + tf.reduce_sum(psi * u, -1))

        sum_log_det_jacob += tf.log(eps + det_jacob)
        z_prev = z_prev + u * h(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b)
############################################################
z_k = z_prev
log_q_k = base_dist.log_prob(z_0) - sum_log_det_jacob
log_p = tf.log(eps + true_density(z_k))

kl = tf.reduce_mean(log_q_k - log_p, -1)

train_op = tf.train.AdamOptimizer(1e-2).minimize(kl)
init_op = tf.global_variables_initializer()


sess = tf.InteractiveSession()
sess.run(init_op)

for i in range(1, 10000 + 1):
    _, kl_np = sess.run([train_op, kl])
    if i % 1000 == 0:
        print('i:', i, 'KL:', kl_np)
    if i % 5000 == 0:
        z_k_np = sess.run(z_k)
        plt.scatter(z_k_np[:, 0], z_k_np[:, 1], alpha=0.7)
        plt.show()