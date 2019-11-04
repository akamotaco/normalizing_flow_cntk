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
def flow_forward(input_dim: int, act_func_pair: tuple = (None, None), batch_norm: bool = False):
    chunk = {}
    log_det_J = 0

    chunk['input_dim'] = input_dim
    _ph = C.placeholder(input_dim, name='place_holder')
    _out = _ph

    if batch_norm:
        # _bn = C.layers.BatchNormalization(name='batch_norm')(_ph)
        # chunk['scale'] = _bn.parameters[0]
        # chunk['bias'] = _bn.parameters[1]

        chunk['mu'] = C.Constant(np.zeros(shape=input_dim))
        chunk['var'] = C.Constant(np.ones(shape=input_dim))

        _eps = C.Constant(1e-7)
        _mu = C.reduce_mean(_ph, axis=C.Axis.default_batch_axis())
        _var = C.reduce_mean(C.square(_ph-_mu), axis=C.Axis.default_batch_axis())

        chunk['muB'] = _mu
        chunk['varB'] = _var

        # _bn = (_ph-chunk['mu'])/C.sqrt(chunk['var']+_eps)
        _bn = C.sqrt(chunk['var']+_eps)*_ph + chunk['mu']
        _ph = _bn

        log_det_J += -0.5*C.reduce_sum(C.log((_var+_eps)))
        # log_det_J += C.reduce_sum(C.log())

    chunk['W_rot_mat'] = _W = C.parameter((input_dim, input_dim))
    _W.value = random_rotation_matrix = special_ortho_group.rvs(input_dim)
    # _W.value = np.roll(np.eye(input_dim),input_dim//2,axis=0)
    _out = _ph@_W
    log_det_J += C.log(C.abs(C.det(_W))) # or # log_det_J += C.slogdet(_W)[1]
    
    _half_dim = input_dim//2
    _x1 = _out[:_half_dim]
    _x2 = _out[_half_dim:]

    _log_s_func, _t_func = act_func_pair
    if _log_s_func is None: # basic network
        _log_s_func = C.layers.Sequential([
            C.layers.Dense(256, C.leaky_relu),
            C.layers.Dense(256, C.leaky_relu),
            C.layers.Dense(_half_dim, C.tanh),
        ])#(C.placeholder(input_dim, name='place_holder'))
    if _t_func is None: # basic network
        _t_func = C.layers.Sequential([
            C.layers.Dense(256, C.leaky_relu),
            C.layers.Dense(256, C.leaky_relu),
            C.layers.Dense(_half_dim),
        ])#(C.placeholder(input_dim, name='place_holder'))

    chunk['log_s_func'] = _log_s_func
    chunk['t_func'] = _t_func

    _log_s, _t = _log_s_func(_x2), _t_func(_x2)

    _s = C.exp(_log_s)

    _y1 = _s*_x1 + _t
    _y2 = _x2

    _Y = C.splice(_y1, _y2)
    chunk['output'] = _Y

    log_det_J += C.reduce_sum(_log_s)

    return _Y, log_det_J, chunk

def flow_reverse(chunk):
    input_dim = chunk['input_dim']
    log_det_J = 0
    _half_dim = input_dim//2

    _ph = C.placeholder(input_dim, name='place_holder')
    _log_s_func = chunk['log_s_func']
    _t_func = chunk['t_func']

    _y1, _y2 = _ph[:_half_dim], _ph[_half_dim:]
    _log_s = _log_s_func(_y2)
    _t = _t_func(_y2)
    _s = C.exp(_log_s)
    _x1 = (_y1-_t)/_s
    _x2 = _y2
    _X = C.splice(_x1, _x2)

    log_det_J += C.reduce_sum(C.log(C.abs(_s)))

    _w = chunk['W_rot_mat']
    chunk['W_rot_mat_inv'] = _inv_w = C.Constant(np.linalg.inv(_w.value), name='inv_W')
    _out = _X@_inv_w
    log_det_J += input_dim*C.log(C.det(_inv_w))

    # if 'scale' in chunk:
    #     _out -= chunk['bias']
    #     _out /= chunk['scale']
    #     log_det_J += input_dim*C.reduce_sum(C.log(C.abs(chunk['scale'])))

    # _out -= chunk['b']
    # _out @= _inv_w

    return _out, log_det_J

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
c_input = C.input_variable(c_dim)

# def _tan(x):
#     return C.tan(x/5)

# def _atan(x):
#     return C.atan(x)*5

# c_block = KLF_forward(c_dim, batch_norm=True)
c_block = []
for i in range(6):
    c_block.append(flow_forward(c_dim, batch_norm=False))


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
    if 'muB' in block[-1]: # batch norm
        bn.append(block[-1]['muB'](q))
        bn.append(block[-1]['varB'](q))
        bn_update.append(block[-1]['mu'])
        bn_update.append(block[-1]['var'])
    q = block[0](q)

base_dist = MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])

# log_q_k = C.log(base_dist.pdf(z_0)) - sum_log_det_jacob

loss = -C.reduce_mean(C.log(base_dist.pdf(q)) + log_det_J)

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


lr_rate = 1e-3
learner = C.adam(loss.parameters, C.learning_parameter_schedule_per_sample(lr_rate), C.momentum_schedule(0.99))
trainer = C.Trainer(loss, (loss, None), [learner])

for i in tqdm(range(10000)):
    # v = np.random.uniform(size=(1,2))
    v = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
    trainer.train_minibatch({loss.arguments[0]:v})

    # from IPython import embed;embed()
    if i%100 == 0:
        print('\n',trainer.previous_minibatch_loss_average)

    if len(bn) > 0: # batch norm
        result = C.combine(bn).eval({loss.arguments[0]:v})
        result = list(result.values())
        momentum = C.Constant(0.9)
        for i in range(len(result)):
            C.assign(bn_update[i],momentum*bn_update[i] + (1-momentum)*result[i]).eval()


import matplotlib.pyplot as plt
# vv = np.random.uniform(size=(1000,2))
vv = noisy_moons
plt.scatter(vv[:,0], vv[:,1]);plt.show()
x = q.eval({q.arguments[0]:vv})
plt.scatter(x[:,0], x[:,1]);plt.show()

plt.hist(x[:,0]);plt.show()
plt.hist(x[:,1]);plt.show()

r_block = []
for i in range(6):
    r_block.append(flow_reverse(c_block[i][-1]))

w = C.input_variable(2)
for i in range(-1,-7,-1):
    w = r_block[i][0](w)

ww = w.eval({w.arguments[0]:x})
www = w.eval({w.arguments[0]:np.random.normal(size=(1000,2))})

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