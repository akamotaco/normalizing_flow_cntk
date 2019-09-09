#%%
import cntk as C
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from tqdm import tqdm

import cntk_expansion
from cntk_distribution import MultivariateNormalDiag


class RealNVP():
    def __init__(self, nets, nett, mask, prior):
        self.prior = prior
        self.mask = mask
        self.t = [nett() for _ in range(mask.shape[0])]
        self.s = [nets() for _ in range(mask.shape[0])]

        self.forward, self.log_det_J = self._forward(2)
        self.reverse = self._reverse(2)
        self.loss = self._loss()

    def _forward(self, input_dim):
        x = C.input_variable(input_dim, needs_gradient=True, name='input')
        z, sum_log_det_jacob = x, C.Constant(0, name='log_det_zero')

        for i in range(len(self.t)):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * C.exp(-s) + z_
            sum_log_det_jacob += C.reduce_sum(s)
        
        z = C.squeeze(z)
        return z, sum_log_det_jacob
    
    def _reverse(self, input_dim):
        x = C.input_variable(input_dim, needs_gradient=True, name='data_input')

        for i in reversed(range(6)):
            x_ = x * self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * C.exp(s) + t)
        x = C.squeeze(x)
        return x
    
    def _loss(self):
        log_q_k = C.log(self.prior.pdf(self.forward)) - self.log_det_J
        return log_q_k
    
    def sample(self, batchSize): 
        z = self.prior.sample(batchSize).astype(np.float32)
        x = self.reverse(z)
        return x

    def parameters(self):
        return self.forward.parameters


nets = lambda: C.layers.Sequential([C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(2, activation=C.tanh)])(C.placeholder(2))
nett = lambda: C.layers.Sequential([C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(2)])(C.placeholder(2))
masks = C.Constant(np.array([[0, 1], [1, 0]] * 3).astype(np.float32), name='mask')
prior = MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
flow = RealNVP(nets, nett, masks, prior)

loss = -C.reduce_mean(flow.loss)

# learner = C.adam(loss.parameters, C.learning_parameter_schedule_per_sample(1e-4), C.momentum_schedule_per_sample(0.9))
learner = C.adam(loss.parameters, C.learning_parameter_schedule(1e-1), C.momentum_schedule(0.9))
trainer = C.Trainer(loss, (loss, None), learner)

for t in range(5001):
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)
    trainer.train_minibatch({loss.arguments[0]:noisy_moons})
    
    if t % 100 == 0:
        print('iter %s:' % t, 'loss = %.3f' % trainer.previous_minibatch_loss_average)

noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
f = flow.forward.eval({flow.forward.arguments[0]:noisy_moons})
plt.scatter(f[:,0],f[:,1])
plt.show()
plt.hist(f[:,0], alpha=0.5)
plt.hist(f[:,1], alpha=0.5)
plt.show()

v = flow.sample(1000)
plt.scatter(v[:,0],v[:,1])
plt.show()
