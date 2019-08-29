#%%
import numpy as np
from scipy.stats import special_ortho_group
import cntk as C
import cntk_expansion
from tqdm import tqdm

#%%
def _linear(x):
    return x

def _leaky_relu(x):
    alpha = 0.001
    _upper = C.greater_equal(x, 0)*x
    _lower = C.less(x, 0)*alpha*x
    return _lower + _upper

def _leaky_relu_inv(x):
    alpha = 1/0.001
    _upper = C.greater_equal(x, 0)*x
    _lower = C.less(x, 0)*alpha*x
    return _lower + _upper

def KLF_forward(input_dim: int, act_func_pair: tuple = (_linear, _linear), batch_norm: bool = False):
    chunk = {}

    chunk['input_dim'] = input_dim
    _ph = C.placeholder(input_dim, name='place_holder')

    _l = C.layers.Dense(input_dim, name='dense')(_ph)
    chunk['W'] = _l.parameters[0]
    chunk['b'] = _l.parameters[1]

    random_rotation_matrix = special_ortho_group.rvs(input_dim)
    chunk['W'].value = random_rotation_matrix

    if batch_norm:
        _bn = C.layers.BatchNormalization(name='batch_norm')(_ph)
        chunk['scale'] = _bn.parameters[0]
        chunk['bias'] = _bn.parameters[1]
        _ph = _bn

    _out = act_func_pair[0](_l)
    chunk['inv_act_func'] = act_func_pair[1]

    return _out, chunk

def KLF_reverse(chunk):
    input_dim = chunk['input_dim']
    _ph = C.placeholder(input_dim, name='place_holder')

    inv_act_func = chunk['inv_act_func']
    _out = inv_act_func(_ph)

    if 'scale' in chunk:
        _out -= chunk['bias']
        _out /= chunk['scale']

    _w = chunk['W']
    _inv_w = C.Constant(np.linalg.inv(_w.value), name='inv_W')

    _out -= chunk['b']
    _out @= _inv_w


    return _out

#%%
# https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

def multivariate_kl_divergence(input_layer):
    _dim = input_layer.shape[0]

    out_value = C.unpack_batch(input_layer)
    _mu1 = C.transpose(C.reduce_mean(out_value, axis=0), [1, 0])
    _sigma1 = C.cov2(input_layer)

    _mu2 = C.zeros_like(_mu1)
    _sigma2 = C.Constant(np.eye(_dim))
    _sigma2_inv = _sigma2 # identity matrix

    return 0.5  * (
        C.log(C.det(_sigma2)/C.det(_sigma1))
        - _dim
        + C.trace(_sigma2_inv@_sigma1)
        + C.transpose((_mu2-_mu1), [1, 0])@_sigma2_inv@(_mu2-_mu1)
        )

#%%
c_dim = 2
c_input = C.input_variable(c_dim, needs_gradient=True)

def _tan(x):
    return C.tan(x/5)

def _atan(x):
    return C.atan(x)*5

# c_block = KLF_forward(c_dim, batch_norm=True)
c_block = []
for i in range(3):
    # c_block.append(KLF_forward(c_dim, batch_norm=True, act_func_pair=(_linear, _linear)))
    # c_block.append(KLF_forward(c_dim, batch_norm=True, act_func_pair=(C.atan, C.tan)))
    c_block.append(KLF_forward(c_dim, batch_norm=True, act_func_pair=(_atan, _tan)))
    # c_block.append(KLF_forward(c_dim, batch_norm=True, act_func_pair=(_leaky_relu, _leaky_relu_inv)))


single = np.array([[1, 2]])
# multi = np.random.uniform(size=(100, 2))
multi = np.random.normal(size=(100, 2))

value = multi.astype(np.float32)

q = c_input
for block in c_block:
    q = block[0](q)
out = q.eval({q.arguments[0]:value})
print(out)

mkld = multivariate_kl_divergence(q)
print(mkld.eval({mkld.arguments[0]:value}))
mkld.grad({mkld.arguments[0]:value})

loss = (mkld)
lr_rate = 0.1
learner = C.adam(q.parameters, C.learning_parameter_schedule(lr_rate), C.momentum_schedule(0.99))
trainer = C.Trainer(q,(loss,None),[learner])

for i in tqdm(range(10000)):
    v = np.random.uniform(size=(1000,2))
    trainer.train_minibatch({q.arguments[0]:v})
    if trainer.previous_minibatch_loss_average < 0:
        print('negative')
    if i%1000 == 0:
        print(trainer.previous_minibatch_loss_average)


# import matplotlib.pyplot as plt
# vv = np.random.uniform(size=(1000,2))
# plt.scatter(vv[:,0], vv[:,1]);plt.show()
# x = q.eval({q.arguments[0]:vv})
# plt.scatter(x[:,0], x[:,1]);plt.show()

# plt.hist(x[:,0]);plt.show()
# plt.hist(x[:,1]);plt.show()

# mm = multivariate_kl_divergence(C.input_variable(2))
# mm.eval({mm.arguments[0]:np.random.normal(size=(1000,2))})
# mm.eval({mm.arguments[0]:x})


# n = np.random.normal(size=(1000,2))
# print(x.mean(axis=0), n.mean(axis=0))
# print(x.std(axis=0), n.std(axis=0))
# print(np.cov(x.T), np.cov(n.T))

# _dim = 2
# _mu1 = x.mean(axis=0)
# _sigma1 = np.cov(x.T)

# _mu2 = np.zeros_like(_mu1)
# _sigma2 = np.eye(_dim)
# _sigma2_inv = _sigma2 # identity matrix

# kld = 0.5  * (
#         np.log(np.linalg.det(_sigma2)/np.linalg.det(_sigma1))
#         - _dim
#         + np.trace(_sigma2_inv@_sigma1)
#         + (_mu2-_mu1).T@_sigma2_inv@(_mu2-_mu1)
#         )

#%%

iq = c_input
for block_inv in reversed(c_block):
    iq = KLF_reverse(block_inv[1])(iq)
inv_out = iq.eval({iq.arguments[0]:out})
print(np.mean((value-inv_out)**2))
iq.eval({iq.arguments[0]:out})
#%%

from IPython import embed;embed()