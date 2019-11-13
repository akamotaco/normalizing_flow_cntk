# https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py

#%%
import numpy as np
from scipy.stats import special_ortho_group
from cntk_distribution import MultivariateNormalDiag
import cntk as C
import cntk_expansion
from tqdm import tqdm

from sklearn import cluster, datasets, mixture
noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)

C.try_set_default_device(C.gpu(0))

class RealNVP():
    @classmethod
    def basic_network(cls, dims: int, op_name: str = '', instance_name: str = ''):
        ph = C.placeholder(dims, name='net_input')
        net = C.layers.Sequential([
                C.layers.Dense(8, C.tanh),
                C.layers.Dense(8, C.tanh),
                C.layers.Dense(dims, name='net_output'),
            ])(ph)
        return C.as_block(net, [(ph,ph)], op_name, instance_name)
    
    @classmethod
    def forward_network(cls, input_dim: int):# , batch_norm: bool = False):
        chunk = {}
        log_det_J = 0

        chunk['input_dim'] = input_dim
        _out = _ph = C.placeholder(input_dim, name='place_holder')

        _half_dim = input_dim//2
        _x1, _x2 = _out[:_half_dim], _out[_half_dim:]

        chunk['log_s_func'] = _log_s_func = cls.basic_network(_half_dim, 'log_s_func')
        chunk['t_func'] = _t_func = cls.basic_network(_half_dim, 't_func')

        _log_s, _t = _log_s_func(_x1), _t_func(_x1)
        _x2 = _t + _x2 * C.exp(_log_s)

        log_det_J += C.reduce_sum(_log_s)

        _out = C.splice(_x1, _x2)

        # ====
        _x1, _x2 = _out[:_half_dim], _out[_half_dim:]

        chunk['log_s_func2'] = _log_s_func2 = cls.basic_network(_half_dim, 'log_s_func2')
        chunk['t_func2'] = _t_func2 = cls.basic_network(_half_dim, 't_func2')

        _log_s2, _t2 = _log_s_func2(_x2), _t_func2(_x2)
        _x1 = _x1 * C.exp(_log_s2) + _t2

        log_det_J += C.reduce_sum(_log_s2)

        _out = _Y = C.splice(_x1, _x2)
        # _out = C.as_block(_out, [(_ph,_ph)],'asdf1','zxcv1')

        return _out, log_det_J, chunk
    
    @classmethod
    def reverse_network(cls, chunk):
        input_dim = chunk['input_dim']
        log_det_J = 0
        _half_dim = input_dim//2

        _out = _ph = C.placeholder(input_dim, name='place_holder')
        _log_s_func, _t_func = chunk['log_s_func'], chunk['t_func']
        _log_s_func2, _t_func2 = chunk['log_s_func2'], chunk['t_func2']

        _y1, _y2 = _ph[:_half_dim], _ph[_half_dim:]

        _log_s2, _t2 = _log_s_func2(_y2), _t_func2(_y2)

         _y1 = (_y1 - _t2) / C.exp(_log_s2)

        log_det_J -= C.reduce_sum(_log_s2)

        # ====

        _log_s, _t = _log_s_func(_y1), _t_func(_y1)
        _y2 = (_y2 - _t) / C.exp(_log_s)

        log_det_J -= C.reduce_sum(_log_s)

        _out = _X = C.splice(_y1, _y2)

        return _out, log_det_J
        
    def __init__(self, dims: int):
        self._forward = self.forward_network(dims)
        self._chunk = self._forward[-1]
        self._reverse = self.reverse_network(self._chunk)
    
    def forward(self, input_: C.Function, sum_log_det_J: C.Function = None):
        log_det_J = self._forward[1](input_)
        output_ = self._forward[0](input_)

        if sum_log_det_J is not None:
            log_det_J += sum_log_det_J

        return output_, log_det_J
    
    def reverse(self, input_: C.Function, sum_log_det_J: C.Function = None):
        log_det_J = self._reverse[1](input_)
        output_ = self._reverse[0](input_)

        if sum_log_det_J is not None:
            log_det_J += sum_log_det_J
        
        return output_, log_det_J
    
    
if __name__ == '__main__':

    #%%
    c_dim = 2
    c_input = C.input_variable(c_dim)

    flow = RealNVP
    flows = [flow(c_dim) for _ in range(1)]

    q = c_input
    log_det_J = 0 # C.Constant(0)
    bn = []
    bn_update = []

    for f in flows:
        q, log_det_J = f.forward(q, log_det_J)

    base_dist = MultivariateNormalDiag(loc=[0.]*c_dim, scale_diag=[1.]*c_dim)

    prior_logprob = base_dist.log_prob(q) # or C.log(base_dist.pdf(q))
    loss = -C.reduce_mean(prior_logprob + log_det_J)

    v = np.r_[np.random.randn(512 // 2, 2) + np.array([5, 3]),
                    np.random.randn(512 // 2, 2) + np.array([-5, 3])]
    v = (v - v.mean(axis=0)) / v.std(axis=0)

    lr_rate = 5e-3
    learner = C.adam(loss.parameters, C.learning_parameter_schedule_per_sample(lr_rate), C.momentum_schedule_per_sample(0.99))

    # lr_rate = 1e-2
    # learner = C.adam(loss.parameters, C.learning_parameter_schedule(lr_rate), C.momentum_schedule(0.99))

    trainer = C.Trainer(loss, (loss, None), [learner])

    for i in tqdm(range(500)):
        # v = np.random.uniform(size=(1000,c_dim))
        # v = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
        # v = np.r_[np.random.randn(512 // 2, 2) + np.array([5, 3]),
        #              np.random.randn(512 // 2, 2) + np.array([-5, 3])]
        out = trainer.train_minibatch({loss.arguments[0]:v}, outputs=[prior_logprob, log_det_J])

        if i%100 == 0:
            logprob = out[1][prior_logprob].mean() + out[1][log_det_J].mean()
            print(f'\n iter:{i} loss:{trainer.previous_minibatch_loss_average:.4} logprob:{logprob:.4} prior:{out[1][prior_logprob].mean():.4} logdet:{out[1][log_det_J].mean():.4}')

    import matplotlib.pyplot as plt
    z = q.eval({q.arguments[0]:v})
    plt.hist(z[:,0],alpha=0.5)
    plt.hist(z[:,1],alpha=0.5)
    plt.savefig('temp.png')

    from IPython import embed;embed()

    w = C.input_variable(2)
    log_det_J = 0
    for f in flows:
        w, log_det_J = f.reverse(w, log_det_J)

    print(f'Reconstruction MSE: {((v - w.eval({w.arguments[0]:q.eval({q.arguments[0]:v})}))**2).mean()}')
    exit()

    # vv = np.random.uniform(size=(1000,c_dim))
    vv = noisy_moons
    plt.scatter(vv[:,0], vv[:,1]);plt.show()
    x = q.eval({q.arguments[0]:vv})
    plt.scatter(x[:,0], x[:,1]);plt.show()

    plt.hist(x[:,0]);plt.show()
    plt.hist(x[:,1]);plt.show()

    w = C.input_variable(c_dim) # reverse flow
    for block in reversed(c_block):
        w = flow_reverse(block[-1])[0](w)

    ww = w.eval({w.arguments[0]:x})
    www = w.eval({w.arguments[0]:np.random.normal(size=(1000,c_dim))})

    plt.scatter(vv[:,0], vv[:,1], alpha=0.5, label='origin')
    plt.scatter(ww[:,0], ww[:,1], alpha=0.5, label='reverse')
    plt.scatter(www[:,0], www[:,1], alpha=0.5, label='generated')
    plt.legend()
    plt.show()

    # mm = multivariate_kl_divergence(C.input_variable(2))
    # mm.eval({mm.arguments[0]:np.random.normal(size=(1000,2))})
    # mm.eval({mm.arguments[0]:x})


    # n = np.random.normal(size=(1000,2))
    # print(x.mean(axis=0), n.mean(axis=0))
    # print(x.std(axis=0), n.std(axis=0))
    # print(np.cov(x.T), np.cov(n.T))

    # # _dim = 2
    # # _mu1 = x.mean(axis=0)
    # # _sigma1 = np.cov(x.T)

    # # _mu2 = np.zeros_like(_mu1)
    # # _sigma2 = np.eye(_dim)
    # # _sigma2_inv = _sigma2 # identity matrix

    # # kld = 0.5  * (
    # #         np.log(np.linalg.det(_sigma2)/np.linalg.det(_sigma1))
    # #         - _dim
    # #         + np.trace(_sigma2_inv@_sigma1)
    # #         + (_mu2-_mu1).T@_sigma2_inv@(_mu2-_mu1)
    # #         )

    # #%%

    # iq = c_input
    # for block_inv in reversed(c_block):
    #     iq = KLF_reverse(block_inv[1])(iq)
    # inv_out = iq.eval({iq.arguments[0]:out})
    # print(np.mean((value-inv_out)**2))
    # iq.eval({iq.arguments[0]:out})
    # #%%
