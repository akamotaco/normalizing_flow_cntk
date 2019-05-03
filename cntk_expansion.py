from autograd import grad, jacobian, elementwise_grad
import autograd.numpy as np
from cntk.ops.functions import UserFunction
from cntk import output_variable

import cntk as C

def __CNTK_dot__(x, y):
    return C.reduce_sum(C.element_times(x,y))
C.dot = __CNTK_dot__


import autograd.numpy.linalg as LA

class MySigmoid(UserFunction):
    def __init__(self, arg, name='MySigmoid'):
        super(MySigmoid, self).__init__([arg], name=name)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.grad = grad(self.sigmoid)

    def forward(self, argument, device=None, outputs_to_retain=None):
        return argument, self.sigmoid(argument)

    def backward(self, state, root_gradients):
        argument = state
        return root_gradients * self.grad(argument)

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype,
            self.inputs[0].dynamic_axes)]

    @staticmethod
    def deserialize(inputs, name, state):
        return MySigmoid(inputs[0], name)

class _CNTK_det_(UserFunction):
    def __init__(self, arg, name='_CNTK_det_'):
        super(_CNTK_det_, self).__init__([arg], name=name)
        self.det = lambda x: LA.det(x)
        self.grad = grad(self.det)

    def forward(self, argument, device=None, outputs_to_retain=None):
        return argument, self.det(argument)

    def backward(self, state, root_gradients):
        argument = state
        return root_gradients * self.grad(argument)

    def infer_outputs(self):
        return [output_variable((), self.inputs[0].dtype,
            self.inputs[0].dynamic_axes)]

    @staticmethod
    def deserialize(inputs, name, state):
        return _CNTK_det_(inputs[0], name)

def __CNTK_det__(X):
    return C.user_function(_CNTK_det_(X))
C.det = __CNTK_det__

from autograd.scipy.stats import multivariate_normal
class _CNTK_mvn_pdf_(UserFunction):
    def __init__(self, X, loc, scale, name='_CNTK_mvn_pdf_'):
        super(_CNTK_mvn_pdf_, self).__init__([X, loc, scale], name=name)
        # def mvn_pdf(X, mu, sig):
        #     sqrt_det_2pi_sig = np.sqrt(2 * np.pi * LA.det(sig))
        #     sig_inv = LA.inv(sig)
        #     X = X[:, None, :] - mu[None, :, :]
        #     return np.exp(-np.matmul(np.matmul(X, np.expand_dims(sig_inv, 0)),
        #                 (X.transpose(0, 2, 1))) / 2) / sqrt_det_2pi_sig
        # self.mvn_pdf = mvn_pdf
        self.mvn_pdf = multivariate_normal.pdf
        # self.grad = grad(self.mvn_pdf)
        self.grad = elementwise_grad(self.mvn_pdf)
        # self.grad = jacobian(self.mvn_pdf)

    def forward(self, arguments, device=None, outputs_to_retain=None):
        X, loc, scale = arguments
        # return None, np.mean(np.zeros_like(X),axis=1).reshape(-1,1,1)
        return arguments, self.mvn_pdf(X, loc, scale).reshape(-1,1).astype(np.float32)

    def backward(self, state, root_gradients, variable):
        X, loc, scale = state
        return root_gradients * self.grad(X, loc, scale)

    def infer_outputs(self):
        # return [output_variable(self.inputs[0].shape, self.inputs[0].dtype,
        return [output_variable((1), self.inputs[0].dtype,
            self.inputs[0].dynamic_axes)]

    # def serialize(self):
        # return {'mvn_pdf' : self.mvn_pdf,
                # 'grad' : self.grad}

    @staticmethod
    def deserialize(inputs, name, state):
        # return _CNTK_mvn_pdf_(inputs[0], name)
        f = _CNTK_mvn_pdf_(inputs[0], inputs[1], inputs[2], name)
        # self.mvn_pdf = state['mnv_pdf']
        # self.grad = state['grad']
        return f

def __CNTK_mvn_pdf__(mu, sig):
    @C.Function
    def _(X): return C.user_function(_CNTK_mvn_pdf_(X, mu,sig))
    return _
C.mvn_pdf = __CNTK_mvn_pdf__

if __name__ == '__main__':
    q = C.mvn_pdf(C.constant([0,0]),C.constant([[1,0],[0,1]]))(C.input_variable(2, needs_gradient=True))

    q.eval({q.arguments[0]:np.random.normal(size=(100,2))})
    q.grad({q.arguments[0]:np.random.normal(size=(100,2))})

    C.det(C.constant([[1,0],[0,1]])).eval()

    C.det(C.constant([[1,0],[0,1]]))
    q = C.det(C.input_variable((2,2),needs_gradient=True))
    q.eval({q.arguments[0]:np.array([[1,0],[0,1]],np.float32)})
    q.grad({q.arguments[0]:np.array([[1,0],[0,1]],np.float32)})