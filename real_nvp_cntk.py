# https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
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

        self.forward, self.log_det_J = self.f(2)
        self.reverse = self.g(2)
        self.log_prob = self._log_prob()

    def f(self, input_dim):
        x = C.input_variable(input_dim, needs_gradient=True, name='input')
        z, sum_log_det_jacob = x, C.Constant(0, name='log_det_zero')

        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = z_ + (1 - self.mask[i]) * (z - t) * C.exp(-s)
            sum_log_det_jacob -= C.reduce_sum(s)

        z = C.squeeze(z)
        return z, sum_log_det_jacob

    def g(self, input_dim):
        x = C.input_variable(input_dim, name='data_input')

        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * C.exp(s) + t)
        x = C.squeeze(x)
        return x

    def _log_prob(self):
        logp = self.log_det_J
        return C.log(self.prior.pdf(self.forward)) + logp

    def sample(self, batchSize): 
        z = self.prior.sample(batchSize).astype(np.float32)
        logp = C.log(self.prior.pdf(z))
        x = self.reverse(z)
        return x

    def parameters(self):
        return self.forward.parameters

if __name__ == '__main__':
    nets = lambda: C.layers.Sequential([C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(2, activation=C.tanh)])(C.placeholder(2))
    nett = lambda: C.layers.Sequential([C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(2)])(C.placeholder(2))
    masks = C.Constant(np.array([[0, 1], [1, 0]] * 3).astype(np.float32), name='mask')
    prior = MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
    flow = RealNVP(nets, nett, masks, prior)

    loss = -C.reduce_mean(flow.log_prob)

    learner = C.adam(loss.parameters, C.learning_parameter_schedule(1e-1), C.momentum_schedule(0.9))
    trainer = C.Trainer(flow.forward, (loss, None), learner)

    for t in range(5001):
        noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
        trainer.train_minibatch({loss.arguments[0]:noisy_moons})

        if t % 500 == 0:
            print('iter %s:' % t, 'loss = %.3f' % trainer.previous_minibatch_loss_average)

    noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
    f = flow.forward.eval({flow.forward.arguments[0]:noisy_moons})

    # v = flow.sample(1000) or
    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000).astype(np.float32)
    r = flow.reverse.eval({flow.reverse.arguments[0]:z})

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2, 2, 1)
    plt.title('two moons')
    plt.scatter(noisy_moons[:, 0], noisy_moons[:, 1])
    plt.subplot(2, 2, 2)
    plt.title('normal flow')
    plt.scatter(f[:, 0], f[:, 1], color='red')
    plt.subplot(2, 2, 3)
    plt.title('reverse flow')
    plt.scatter(r[:, 0], r[:, 1])
    plt.subplot(2, 2, 4)
    plt.title('normal distribution')
    plt.scatter(z[:, 0], z[:, 1], color='red')
    plt.savefig('./reports/real_nvp_two_moons.png')