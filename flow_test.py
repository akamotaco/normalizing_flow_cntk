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

#%%
def flow_forward(input_dim: int, act_func_pair: tuple = (None, None)):# , batch_norm: bool = False):
    chunk = {}
    log_det_J = 0

    chunk['input_dim'] = input_dim
    _ph = C.placeholder(input_dim, name='place_holder')
    _out = _ph

    #================================================================

    _half_dim = input_dim//2
    _x1 = _out[:_half_dim] # lower
    _x2 = _out[_half_dim:] # upper
    lower, upper = _x1, _x2

    _log_s_func, _t_func = None, None # act_func_pair
    if _log_s_func is None: # basic network
        _log_s_func = C.layers.Sequential([
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(_half_dim, None),
        ])
    if _t_func is None: # basic network
        _t_func = C.layers.Sequential([
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(_half_dim, None),
        ])

    _log_s_func2, _t_func2 = None, None
    if _log_s_func2 is None: # basic network
        _log_s_func2 = C.layers.Sequential([
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(_half_dim,None),
        ])
    if _t_func2 is None: # basic network
        _t_func2 = C.layers.Sequential([
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(8, C.tanh),
            C.layers.Dense(_half_dim,None),
        ])

    # chunk['log_s_func'] = _log_s_func
    # chunk['t_func'] = _t_func

    _log_s, _t = _log_s_func(lower), _t_func(lower)
    upper = _t + upper * C.exp(_log_s)
    _log_s2, _t2 = _log_s_func2(upper), _t_func2(upper)
    lower = _t2 + lower * C.exp(_log_s2)
    _out = C.splice(lower,upper)

    log_det_J += C.reduce_sum(_log_s) + C.reduce_sum(_log_s2)
    

    # log_det_J += C.reduce_sum(_log_s)

    #========================================================================

    # chunk['W_rot_mat'] = _W = C.Constant(np.roll(np.eye(input_dim),input_dim//2,axis=0))
    # _out = _out@_W

    return _out, log_det_J, chunk

# def flow_reverse(chunk):
#     input_dim = chunk['input_dim']
#     log_det_J = 0
#     _half_dim = input_dim//2

#     _ph = C.placeholder(input_dim, name='place_holder')
#     _log_s_func = chunk['log_s_func']
#     _t_func = chunk['t_func']

#     _y1, _y2 = _ph[:_half_dim], _ph[_half_dim:]
#     _log_s = _log_s_func(_y2)
#     _t = _t_func(_y2)
#     _s = C.exp(_log_s)
#     _x1 = (_y1-_t)/_s
#     _x2 = _y2
#     _X = C.splice(_x1, _x2)

#     log_det_J += C.reduce_sum(C.log(C.abs(_s)))

#     _w = chunk['W_rot_mat']
#     chunk['W_rot_mat_inv'] = _inv_w = C.Constant(np.linalg.inv(_w.value), name='inv_W')
#     _out = _X@_inv_w
#     log_det_J += input_dim*C.log(C.det(_inv_w))

#     # if 'scale' in chunk:
#     #     _out -= chunk['bias']
#     #     _out /= chunk['scale']
#     #     log_det_J += input_dim*C.reduce_sum(C.log(C.abs(chunk['scale'])))

#     # _out -= chunk['b']
#     # _out @= _inv_w

#     return _out, log_det_J

#%%
c_dim = 2
c_input = C.input_variable(c_dim)

# def _tan(x):
#     return C.tan(x/5)

# def _atan(x):
#     return C.atan(x)*5

# c_block = KLF_forward(c_dim, batch_norm=True)
c_block = []
for i in range(1):
    c_block.append(flow_forward(c_dim))


# single = np.array([[1, 2]])
# # multi = np.random.uniform(size=(100, 2))
# multi = np.random.normal(size=(100, 2))

# value = multi.astype(np.float32)

q = c_input
log_det_J = C.Constant(0)
bn = []
bn_update = []
for block in c_block:
    log_det_J += block[1](q)
    q = block[0](q)

base_dist = MultivariateNormalDiag(loc=[0.]*c_dim, scale_diag=[1.]*c_dim)

# log_q_k = C.log(base_dist.pdf(z_0)) - sum_log_det_jacob

prior_logprob = base_dist.log_prob(q) #C.log(base_dist.pdf(q))
# prior_logprob = C.log(base_dist.pdf(q))
loss = -C.reduce_mean(prior_logprob + log_det_J)

# mkld = multivariate_kl_divergence(q)
# print(mkld.eval({mkld.arguments[0]:value}))
# mkld.grad({mkld.arguments[0]:value})1
# loss = (mkld)

# _q_prime = C.tanh(q)
# _mu = C.reduce_mean(_q_prime, axis=C.Axis.default_batch_axis())
# _sigma = C.reduce_mean(C.square(_q_prime-_mu), axis=C.Axis.default_batch_axis())
# loss += C.reduce_mean(C.square(_mu)) + C.reduce_mean(C.square(_sigma-0.615))

# # _log_mu = C.reduce_mean(C.log(C.abs(q)), axis=C.Axis.default_batch_axis())
# # loss += C.reduce_mean(C.square(_log_mu+0.57))

from IPython import embed;embed()
exit()


lr_rate = 5e-4 # 1e-2
learner = C.adam(loss.parameters, C.learning_parameter_schedule_per_sample(lr_rate), C.momentum_schedule_per_sample(0.99))
# learner = C.adam(loss.parameters, C.learning_parameter_schedule(lr_rate), C.momentum_schedule(0.99))
trainer = C.Trainer(loss, (loss, None), [learner])

v = np.r_[np.random.randn(512 // 2, 2) + np.array([5, 3]),
                np.random.randn(512 // 2, 2) + np.array([-5, 3])]
# v[:,i] = (v[:,i] - np.mean(v[:,i])) / np.std(v[:,i])
v = (v - v.mean(axis=0)) / v.std(axis=0)

for i in tqdm(range(500)):
    # v = np.random.uniform(size=(1000,c_dim))
    # v = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
    # v = np.r_[np.random.randn(512 // 2, 2) + np.array([5, 3]),
    #              np.random.randn(512 // 2, 2) + np.array([-5, 3])]
    out = trainer.train_minibatch({loss.arguments[0]:v}, outputs=[prior_logprob, log_det_J])

    # from IPython import embed;embed()
    if i%100 == 0:
        logprob = out[1][prior_logprob].mean() + out[1][log_det_J].mean()
        print(f'\n iter:{i} loss:{trainer.previous_minibatch_loss_average} logprob:{logprob} prior:{out[1][prior_logprob].mean()} logdet:{out[1][log_det_J].mean()}')

import matplotlib.pyplot as plt
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

from IPython import embed;embed()