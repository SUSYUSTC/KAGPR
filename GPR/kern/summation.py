import typing as tp
import numpy as np
from .kernel import Kernel
from .cache import Cache
from . import get_kern_obj
from .. import utils


class Summation(Kernel):
    def __init__(self, kernel: Kernel):
        self.name = 'summation.Summation'
        self.default_cache = {}
        self.kernel = kernel
        self.ps = self.kernel.ps
        self.ps_bound = self.kernel.ps_bound
        self.transformations = self.kernel.transformations
        self.nout = self.kernel.nout
        self.dK_dps = []
        for i in range(len(self.kernel.ps)):
            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp(i, X, X2, **kwargs)
            self.dK_dps.append(func)
        super().__init__()
        self.check()

    def sum_by_length(self, K, lengths1, lengths2):
        assert K.shape == (np.sum(lengths1), np.sum(lengths2))
        xp = utils.get_array_module(K)
        indices1 = xp.cumsum(xp.array(lengths1))
        indices2 = xp.cumsum(xp.array(lengths2))
        K = xp.diff(xp.cumsum(K, axis=0)[indices1 - 1, :], prepend=0, axis=0)
        K = xp.diff(xp.cumsum(K, axis=1)[:, indices2 - 1], prepend=0, axis=1)
        return K

    def _fake_K(self, method, X, X2=None):
        if X2 is None:
            X2 = X
        xp = utils.get_array_module(X)
        lengths1 = np.array([len(item) for item in X])
        lengths2 = np.array([len(item) for item in X2])
        K = method(xp.concatenate(X), xp.concatenate(X2))
        return self.sum_by_length(K, lengths1, lengths2)

    @Cache('g')
    def K(self, X, X2=None):
        return self._fake_K(self.kernel.K, X, X2)

    @Cache('g')
    def dK_dp(self, i, X, X2=None):
        return self._fake_K(self.kernel.dK_dps[i], X, X2)

    def clear_cache(self) -> None:
        self.cache_data = {}
        self.kernel.clear_cache()

    def set_cache_state(self, state: bool) -> None:
        self.cache_state = state
        self.kernel.set_cache_state(state)

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        data = {
            'name': self.name,
            'kern': self.kernel.to_dict(),
        }
        return data

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> Kernel:
        kern_dict = data['kern']
        kernel = get_kern_obj(kern_dict)
        result = self(kernel)
        return result
