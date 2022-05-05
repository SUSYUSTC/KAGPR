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
        self.save_on_CPU = False
        self.ps = self.kernel.ps
        self.set_ps = self.kernel.set_ps
        self.transformations = self.kernel.transformations
        self.nout = self.kernel.nout
        self.dK_dps = []
        for i in range(len(self.kernel.ps)):
            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp(i, X, X2, **kwargs)
            self.dK_dps.append(func)
        self.set_onetime_number()
        super().__init__()
        self.check()

    def set_save_on_CPU(self, save_on_CPU: bool):
        self.save_on_CPU = save_on_CPU

    def set_onetime_number(self, n=5000):
        self.onetime_number = n

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
        xp = utils.get_array_module(X[0])
        N1 = len(X)
        N2 = len(X2)
        lengths1 = np.array([len(item) for item in X])
        lengths2 = np.array([len(item) for item in X2])
        split1 = utils.split_by_onetime_number(X, self.onetime_number)
        split2 = utils.split_by_onetime_number(X2, self.onetime_number)
        if self.save_on_CPU:
            result = np.zeros((N1, N2))
        else:
            result = xp.zeros((N1, N2))
        for slic1 in split1:
            for slic2 in split2:
                K = method(xp.concatenate(X[slic1]), xp.concatenate(X2[slic2]))
                tmp = self.sum_by_length(K, lengths1[slic1], lengths2[slic2])
                if self.save_on_CPU:
                    tmp = xp.asnumpy(tmp)
                result[slic1, slic2] = tmp
        return result

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
