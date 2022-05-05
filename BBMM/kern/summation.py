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

    def _fake_K(self, method, X, X2=None, save_on_CPU=False, nGPUs=None):
        if X2 is None:
            X2 = X
        xp = utils.get_array_module(X[0])
        N1 = len(X)
        N2 = len(X2)
        lengths1 = np.array([len(item) for item in X])
        lengths2 = np.array([len(item) for item in X2])
        split1 = utils.split_by_onetime_number(X, self.onetime_number)
        split2 = utils.split_by_onetime_number(X2, self.onetime_number)
        if save_on_CPU:
            result = np.zeros((N1, N2))
        else:
            result = xp.zeros((N1, N2))
        index = 0
        if nGPUs is not None:
            X_devices = [None for i in range(nGPUs)]
            X2_devices = [None for i in range(nGPUs)]
            for i in range(nGPUs):
                with xp.cuda.Device(i):
                    X_devices[i] = [xp.asarray(item) for item in X]
                    X2_devices[i] = [xp.asarray(item) for item in X2]
        s1 = len(split1)
        s2 = len(split2)
        values = [[None for i in range(s2)] for i in range(s1)]
        for i, slic1 in enumerate(split1):
            for j, slic2 in enumerate(split2):
                if nGPUs is not None:
                    with xp.cuda.Device(index):
                        subX = xp.concatenate(X_devices[index][slic1])
                        subX2 = xp.concatenate(X2_devices[index][slic2])
                        print(subX.shape, subX2.shape, index)
                        K = method(subX, subX2)
                        values[i][j] = self.sum_by_length(K, lengths1[slic1], lengths2[slic2])
                else:
                    K = method(xp.concatenate(X[slic1]), xp.concatenate(X2[slic2]))
                    values[i][j] = self.sum_by_length(K, lengths1[slic1], lengths2[slic2])
                if nGPUs is not None:
                    index = (index + 1) % nGPUs
        for i, slic1 in enumerate(split1):
            for j, slic2 in enumerate(split2):
                tmp = values[i][j]
                if save_on_CPU:
                    tmp = xp.asnumpy(tmp)
                elif not isinstance(tmp, np.ndarray):
                    with xp.cuda.Device(i):
                        tmp = xp.asarray(tmp)
                result[slic1, slic2] = tmp
        return result

    @Cache('g')
    def K(self, X, X2=None, save_on_CPU=False, nGPUs=None):
        return self._fake_K(self.kernel.K, X, X2, save_on_CPU=save_on_CPU, nGPUs=nGPUs)

    @Cache('g')
    def dK_dp(self, i, X, X2=None, save_on_CPU=False, nGPUs=None):
        return self._fake_K(self.kernel.dK_dps[i], X, X2, save_on_CPU=save_on_CPU, nGPUs=nGPUs)

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
