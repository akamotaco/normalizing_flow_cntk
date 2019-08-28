#%%
import numpy as np
import cntk as C

#%%
def _linear(x):
    return x

def KLF_forward(input_shape: tuple, act_func_pair: tuple = (_linear, _linear), batch_norm: bool = False):
    chunk = {}

    chunk['input_shape'] = input_shape
    _ph = C.placeholder(input_shape, name='place_holder')

    _l = C.layers.Dense(input_shape, name='dense')(_ph)
    chunk['W'] = _l.parameters[0]
    chunk['b'] = _l.parameters[1]

    if batch_norm:
        _bn = C.layers.BatchNormalization(name='batch_norm')(_ph)
        chunk['scale'] = _bn.parameters[0]
        chunk['bias'] = _bn.parameters[1]
        _ph = _bn

    _out = act_func_pair[0](_l)
    chunk['inv_act_func'] = act_func_pair[1]

    return _out, chunk

def KLF_reverse(chunk):
    input_shape = chunk['input_shape']
    _ph = C.placeholder(input_shape, name='place_holder')

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
c_shape = (2)
c_input = C.input_variable(c_shape)

c_block = KLF_forward(c_shape, batch_norm=True)



q = c_block[0](c_input)
out = q.eval({q.arguments[0]:np.array([[1, 2]])})
print(out)


#%%

c_inv_block = KLF_reverse(c_block[1])
iq = c_inv_block(c_input)
inv_out = iq.eval({iq.arguments[0]:out})
print(out)
print(inv_out)
#%%
