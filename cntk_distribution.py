import cntk as C
import numpy as np

import cntk_expansion


class MultivariateNormalDiag():
    def __init__(self, loc, scale_diag):
        self.loc = np.array(loc)
        self.scale = np.array(scale_diag) * np.eye(self.loc.shape[0])

        self.loc, self.scale = self.loc.astype(np.float32), self.scale.astype(np.float32)
        self.shape = self.loc.shape
        self.mvn_pdf = C.mvn_pdf(C.constant(self.loc, name='loc'),
                                 C.constant(self.scale, name='scale'))
    def size(self):
        return self.loc.shape
    def sample(self, count):
        return np.random.multivariate_normal(self.loc, self.scale, count)
    def pdf(self, x):
        return self.mvn_pdf(x)
