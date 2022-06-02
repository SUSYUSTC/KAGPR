import typing as tp
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#import numpy as np
from .kernel import Kernel
from .cache import Cache
from .param import Param
from . import param_transformation


class Linear(Kernel):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'polynomial.Linear'
        self.default_cache = {'g': 0}
        self.dK_dps = []
        self.ps = []
        self.nout = 1
        self.transformations = []
        self.check()

    def clear_cache(self):
        self.cache_data = {}

    @Cache('g')
    def K(self, X, X2=None):
        if X2 is None:
            return X.dot(X.T)
        else:
            return X.dot(X2.T)

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        data = {
            'name': self.name
        }
        return data

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> Kernel:
        kernel = self()
        return kernel

    def set_cache_state(self, state):
        self.cache_statue = state


class Polynomial(Kernel):
    def __init__(self, order, opt=False) -> None:
        super().__init__()
        self.name = 'polynomial.Polynomial'
        self.default_cache = {}
        self.order = Param('order', order)
        self.nout = 1
        self.opt = opt
        if self.opt:
            self.dK_dps = [self.dK_dorder]
            self.ps = [self.order]
            self.transformations = [param_transformation.linear]
            self.ps_bound = [(0, np.inf)]
        else:
            self.dK_dps = []
            self.ps = []
            self.transformations = []
            self.ps_bound = []
        self.check()

    def set_order(self, order):
        self.order.value = float(order)

    def clear_cache(self):
        self.cache_data = {}

    @Cache('g')
    def K(self, X, X2=None):
        order = self.order.value
        if X2 is None:
            tmp = X**order
            return tmp.dot(tmp.T)
        else:
            return (X**order).dot((X2**order).T)

    @Cache('g')
    def dK_dorder(self, X, X2=None):
        order = self.order.value
        if X2 is None:
            tmp = X**order
            tmplog = tmp * np.log(X)
            part = tmp.dot(tmplog.T)
            return part + part.T
        else:
            part1 = (X**order * np.log(X)).dot((X2**order).T)
            part2 = (X**order).dot(((X2**order) * np.log(X2)).T)
            return part1 + part2

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        data = {
            'name': self.name,
            'order': self.order,
        }
        return data

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> Kernel:
        kernel = self()
        kernel.set_order(data['order'])
        return kernel

    def set_cache_state(self, state):
        self.cache_statue = state
