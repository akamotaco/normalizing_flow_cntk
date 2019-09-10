#%%
import cntk as C
import matplotlib.pyplot as plt
import numpy as np

import cntk_expansion
from cntk_distribution import MultivariateNormalDiag

#%%
K = 16
EPS = 1e-7

#%%
def true_density(z):
    z1, z2 = z[0], z[1]
    norm = C.sqrt(C.square(z1) + C.square(z2))
    exp1 = C.exp(-0.5 * C.square((z1 - 2) / 0.8))
    exp2 = C.exp(-0.5 * C.square((z1 + 2) / 0.8))
    u = 0.5 * C.square(((norm - 4) / 0.4)) - C.log(exp1 + exp2)
    return C.exp(-u)

#%%
h = lambda x: C.tanh(x)
h_prime = lambda x: 1 - C.square(C.tanh(x))

base_dist = MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
z_0 = C.input_variable(base_dist.size(), name='sampled')
z_prev = z_0
sum_log_det_jacob = 0.

initializer = C.initializer.uniform(1)
for i in range(K):
    u = C.parameter((2), name='u', init=initializer)
    w = C.parameter((2), name='w', init=initializer)
    b = C.parameter((1), name='b', init=initializer)

    psi = h_prime(C.dot(w, z_prev)+b) * w
    det_jacob = C.abs(1 + C.dot(u, psi))

    sum_log_det_jacob += C.log(EPS + det_jacob)
    z_prev = z_prev + u * h(C.dot(w, z_prev)+b)

z_k = z_prev
log_q_k = C.log(base_dist.pdf(z_0)) - sum_log_det_jacob
log_p = C.log(EPS + true_density(z_k))

kl = C.reduce_mean(log_q_k - log_p)
#%%
lr = 10
lr_schedule = C.learning_parameter_schedule(lr)
learner = C.adam(kl.parameters, lr_schedule, 0.9)
trainer = C.Trainer(kl, (kl, None), learner)

#%%
for i in range(1, 2000 + 1):
    s = base_dist.sample(500).astype(np.float32)
    trainer.train_minibatch({kl.arguments[0]:s})
    if i % 100 == 0:
        print(trainer.previous_minibatch_loss_average)
    # if i % 500 == 0:
    #     v = z_k.eval({z_k.arguments[0]:s})
    #     plt.scatter(v[:, 0], v[:, 1], alpha=0.7)
    #     plt.show()

v = z_k.eval({z_k.arguments[0]:s})
plt.scatter(v[:, 0], v[:, 1], alpha=0.5, c='green')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title('Flow Length (K): %d' % K)
plt.savefig('./reports/example1-k%d.png' % K, dpi=300)
