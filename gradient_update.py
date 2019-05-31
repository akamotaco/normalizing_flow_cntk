#%%
import cntk as C
import numpy as np

a = C.input_variable(1,needs_gradient=True)
b = C.constant(2)
c = C.square(a-b)
loss = C.reduce_mean(c)

#%%
v = np.array([0],np.float32)
g = loss.grad({a:v})

#%%
lr = 0.11
maximum = 100
eps = 1e-12
for i in range(maximum):
    g = loss.grad({a:v})[0]
    old_v = v
    v -= g*lr
    print(f'v:{old_v}->{v}')

    last = loss.eval({a:v})
    if last < eps:
        print(f'last:{last}')
        break

#%%
