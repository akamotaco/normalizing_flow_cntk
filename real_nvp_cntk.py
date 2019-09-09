#%%
import numpy as np
from scipy.stats import special_ortho_group
from cntk_distribution import MultivariateNormalDiag
import cntk as C
import cntk_expansion
from tqdm import tqdm

from sklearn import cluster, datasets, mixture
noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)

class RealNVP():
    def __init__(self, nets, nett, mask, prior):
        self.prior = prior
        # self.mask = nn.Parameter(mask, requires_grad=False)
        self.mask = mask
        self.t = [nett(C.placeholder(2)) for _ in range(mask.shape[0])]
        self.s = [nets(C.placeholder(2)) for _ in range(mask.shape[0])]
        # self.p = []

        self.forward, self.log_det_J = self._forward(2)
        # self.p += self.forward.parameters

        self.loss = self._loss()

    def _forward(self, input_dim):
        x = C.input_variable(input_dim, needs_gradient=True)
        z, log_det_J = x, C.Constant(0)

        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * C.exp(-s) + z_
            log_det_J -= C.reduce_sum(s)
        from IPython import embed;embed()
        return z, log_det_J
    
    def _loss(self):
        return C.log(self.prior.pdf(C.squeeze(self.forward))) + self.log_det_J
        
    # def g(self, z):
    #     x = z
    #     for i in range(len(self.t)):
    #         x_ = x*self.mask[i]
    #         s = self.s[i](x_)*(1 - self.mask[i])
    #         t = self.t[i](x_)*(1 - self.mask[i])
    #         x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
    #     return x

    # def f(self, x):
    #     log_det_J, z = x.new_zeros(x.shape[0]), x
    #     for i in reversed(range(len(self.t))):
    #         z_ = self.mask[i] * z
    #         s = self.s[i](z_) * (1-self.mask[i])
    #         t = self.t[i](z_) * (1-self.mask[i])
    #         z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
    #         log_det_J -= s.sum(dim=1)
    #     return z, log_det_J
    
    # def log_prob(self,x):
    #     z, logp = self.f(x)
    #     return self.prior.log_prob(z) + logp
        
    # def sample(self, batchSize): 
    #     z = self.prior.sample((batchSize, 1))
    #     logp = self.prior.log_prob(z)
    #     x = self.g(z)
    #     return x
    
    # def parameters(self):
    #     return self.p



# nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
# nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
nets = lambda x: C.layers.Sequential([C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(2, activation=C.tanh)])(x)
nett = lambda x: C.layers.Sequential([C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(256, activation=C.leaky_relu), C.layers.Dense(2)])(x)
masks = C.Constant(np.array([[0, 1], [1, 0]] * 3).astype(np.float32), name='mask')
prior = MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
flow = RealNVP(nets, nett, masks, prior)


# optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)
loss = C.reduce_mean(-flow.loss)
learner = C.adam(loss.parameters, C.learning_parameter_schedule(1e-4), C.momentum_schedule(0.99))
trainer = C.Trainer(loss, (loss, None), learner)
from IPython import embed;embed()
for t in range(5001):
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)
    trainer.train_minibatch({loss.arguments[0]:noisy_moons})
    
    if t % 500 == 0:
        print('iter %s:' % t, 'loss = %.3f' % trainer.previous_minibatch_loss_average)

from IPython import embed;embed()