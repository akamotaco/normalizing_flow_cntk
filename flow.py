#%%
import numpy as np
from scipy.stats import special_ortho_group
from cntk_distribution import MultivariateNormalDiag
import cntk as C
import cntk_expansion
from tqdm import tqdm

from sklearn import cluster, datasets, mixture
noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)

#%%
def flow_forward(input_dim: int, act_func_pair: tuple = (None, None), batch_norm: bool = False):
    chunk = {}
    log_det_J = 0

    _a = C.placeholder(input_dim, name='place_holder')
    _b =  C.parameter(2)
    _c = _b*_a

    return _c, log_det_J, chunk

#     chunk['input_dim'] = input_dim
#     _ph = C.placeholder(input_dim, name='place_holder')


#     if batch_norm:
#         _bn = C.layers.BatchNormalization(name='batch_norm')(_ph)
#         chunk['scale'] = _bn.parameters[0]
#         chunk['bias'] = _bn.parameters[1]
#         _ph = _bn
#         # log_det_J += input_dim*C.reduce_sum(C.log(C.abs(chunk['scale'])))

#     chunk['W_rot_mat'] = _W = C.parameter((input_dim, input_dim))
#     _W.value = random_rotation_matrix = special_ortho_group.rvs(input_dim)
#     _out = _ph@_W
#     # log_det_J += input_dim*C.log(C.abs(C.det(_W)))

#     _half_dim = input_dim//2
#     _x1 = _out[:_half_dim]
#     _x2 = _out[_half_dim]

#     _log_s_func, _t_func = act_func_pair
#     if _log_s_func is None: # basic network
#         _log_s_func = C.layers.Sequential([
#             C.layers.Dense(256, C.leaky_relu),
#             C.layers.Dense(256, C.leaky_relu),
#             C.layers.Dense(input_dim//2, C.tanh),
#         ])#(C.placeholder(input_dim, name='place_holder'))
#     if _t_func is None: # basic network
#         _t_func = C.layers.Sequential([
#             C.layers.Dense(256, C.leaky_relu),
#             C.layers.Dense(256, C.leaky_relu),
#             C.layers.Dense(input_dim//2),
#         ])#(C.placeholder(input_dim, name='place_holder'))

#     chunk['log_s_func'] = _log_s_func
#     chunk['t_func'] = _t_func

#     _log_s, _t = _log_s_func(_x2), _t_func(_x2)

#     _s = C.exp(_log_s)

#     _y1 = _x1*_s + _t
#     _y2 = _x2

#     _Y = C.splice(_y1, _y2)
#     chunk['output'] = _Y

#     # log_det_J += C.reduce_sum(C.log(C.abs(_s)))

#     return _Y, log_det_J, chunk

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
#     _out = _inv_w@_X
#     log_det_J += input_dim*C.log(C.det(_inv_w))

#     if 'scale' in chunk:
#         _out -= chunk['bias']
#         _out /= chunk['scale']
#         log_det_J += input_dim*C.reduce_sum(C.log(C.abs(chunk['scale'])))

#     _out -= chunk['b']
#     _out @= _inv_w

#     return _out, log_det_J

#%%
# https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

# def multivariate_kl_divergence(input_layer):
#     _dim = input_layer.shape[0]

#     out_value = C.unpack_batch(input_layer)
#     _mu1 = C.transpose(C.reduce_mean(out_value, axis=0), [1, 0])
#     _sigma1 = C.cov2(input_layer)

#     _mu2 = C.zeros_like(_mu1)
#     _sigma2 = C.Constant(np.eye(_dim))
#     _sigma2_inv = _sigma2 # identity matrix

#     return 0.5  * (
#         C.log(C.det(_sigma2)/C.det(_sigma1))
#         - _dim
#         + C.trace(_sigma2_inv@_sigma1)
#         + C.transpose((_mu2-_mu1), [1, 0])@_sigma2_inv@(_mu2-_mu1)
#         )

#%%
c_dim = 2
c_input = C.input_variable(c_dim, needs_gradient=True)

a = C.parameter(2)
a.value = np.array([1,2])

q = c_input*a

# def _tan(x):
#     return C.tan(x/5)

# def _atan(x):
#     return C.atan(x)*5

# # c_block = KLF_forward(c_dim, batch_norm=True)
# c_block = []
# for i in range(1):
#     c_block.append(flow_forward(c_dim, batch_norm=False))

# single = np.array([[1, 2]])
# # multi = np.random.uniform(size=(100, 2))
# multi = np.random.normal(size=(100, 2))

# value = multi.astype(np.float32)

# q = c_input
# log_det_J = C.zeros_like(c_dim)
# for block in c_block:
    # log_det_J += block[1](q)
    # q = block[0](q)
# out = q.eval({q.arguments[0]:value})
# print(out)

base_dist = MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])

# log_q_k = C.log(base_dist.pdf(z_0)) - sum_log_det_jacob

loss = C.reduce_mean(C.log(base_dist.pdf(c_input))) # - log_det_J)

# mkld = multivariate_kl_divergence(q)
# print(mkld.eval({mkld.arguments[0]:value}))
# mkld.grad({mkld.arguments[0]:value})
# loss = (mkld)

# _q_prime = C.tanh(q)
# _mu = C.reduce_mean(_q_prime, axis=C.Axis.default_batch_axis())
# _sigma = C.reduce_mean(C.square(_q_prime-_mu), axis=C.Axis.default_batch_axis())
# loss += C.reduce_mean(C.square(_mu)) + C.reduce_mean(C.square(_sigma-0.615))

# # _log_mu = C.reduce_mean(C.log(C.abs(q)), axis=C.Axis.default_batch_axis())
# # loss += C.reduce_mean(C.square(_log_mu+0.57))



lr_rate = 0.01
learner = C.adam(q.parameters, C.learning_parameter_schedule(lr_rate), C.momentum_schedule(0.99))
trainer = C.Trainer(q,(loss,None),[learner])

for i in tqdm(range(10)):
    v = np.random.uniform(size=(1,2))
    # v = noisy_moons
    trainer.train_minibatch({q.arguments[0]:v})
    # if i%100 == 0:
        # print(trainer.previous_minibatch_loss_average)

import matplotlib.pyplot as plt
vv = np.random.uniform(size=(1000,2))
plt.scatter(vv[:,0], vv[:,1]);plt.show()
x = q.eval({q.arguments[0]:vv})
plt.scatter(x[:,0], x[:,1]);plt.show()

# plt.hist(x[:,0]);plt.show()
# plt.hist(x[:,1]);plt.show()

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