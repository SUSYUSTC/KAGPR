import typing as tp
import functools
import numpy as np
from .kernel import Kernel
from .cache import Cache
from . import get_kern_obj
from .. import utils

type_dims = tp.List[tp.Union[slice, np.ndarray]]


def add(x, y):
    return x + y


def concatenate(ls):
    return functools.reduce(add, ls)


class ProductKernel(Kernel):
    def __init__(self, kern_list: tp.List[Kernel], dims: type_dims = None) -> None:
        self.name = 'kern_operation.ProductKernel'
        self.nk = len(kern_list)
        self.kern_list = kern_list
        assert np.all(np.array([k.nout for k in kern_list]) == kern_list[0].nout)
        self.nout = kern_list[0].nout
        self.nps = [len(k.ps) for k in self.kern_list]
        self.cumsum = np.concatenate([np.array([0]), np.cumsum(self.nps)])
        self.cache_K: tp.Dict[str, tp.Any] = {}
        self.cache_dK_dp: tp.Dict[str, tp.Any] = {}
        self.ps = concatenate([k.ps for k in self.kern_list])
        self.ps_bound = concatenate([k.ps_bound for k in self.kern_list])
        self.dK_dps = []
        self.transformations = concatenate([k.transformations for k in self.kern_list])
        self.default_cache: tp.Dict[str, tp.Any] = {}
        if dims is None:
            self.dims: type_dims = [slice(None, None, None) for i in range(self.nk)]
        else:
            self.dims = dims
            assert len(self.dims) == self.nk
        for i in range(len(self.ps)):
            kern_index, pos_index = self.get_pos(i)

            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp(i, X, X2, **kwargs)
            self.dK_dps.append(func)
        super().__init__()
        self.check()

    def get_pos(self, i: int) -> tp.Tuple[int, int]:
        kern_index = len(np.where(self.cumsum <= i)[0]) - 1
        pos_index = i - self.cumsum[kern_index]
        return (kern_index, pos_index)

    def cached_K(self, kern_index, X, X2=None):
        X = X[:, self.dims[kern_index]]
        if X2 is not None:
            X2 = X2[:, self.dims[kern_index]]
        if kern_index not in self.cache_K:
            self.cache_K[kern_index] = self.kern_list[kern_index].K(X, X2)
        return self.cache_K[kern_index]

    def cached_dK_dp(self, kern_index, pos_index, X, X2=None):
        X = X[:, self.dims[kern_index]]
        if X2 is not None:
            X2 = X2[:, self.dims[kern_index]]
        if (kern_index, pos_index) not in self.cache_dK_dp:
            self.cache_dK_dp[(kern_index, pos_index)] = self.kern_list[kern_index].dK_dps[pos_index](X, X2)
        return self.cache_dK_dp[(kern_index, pos_index)]

    @Cache('g')
    def K(self, X, X2=None):
        xp = utils.get_array_module(X)
        Ks = [self.cached_K(i, X, X2) for i in range(self.nk)]
        if not self.cache_state:
            self.clear_cache()
        return functools.reduce(xp.multiply, Ks)

    @Cache('g')
    def dK_dp(self, i, X, X2=None):
        xp = utils.get_array_module(X)
        kern_index, pos_index = self.get_pos(i)
        Ks = [self.cached_K(j, X, X2) for j in range(self.nk)]
        Ks[kern_index] = self.cached_dK_dp(kern_index, pos_index, X, X2)
        self.Ks = Ks
        if not self.cache_state:
            self.clear_cache()
        return functools.reduce(xp.multiply, Ks)

    def clear_cache(self) -> None:
        self.cache_K = {}
        self.cache_dK_dp = {}
        for i in range(self.nk):
            self.kern_list[i].clear_cache()

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        data = {
            'kern_list': [k.to_dict() for k in self.kern_list],
            'dims': self.dims,
            'name': self.name,
        }
        return data

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> Kernel:
        kern_list = [get_kern_obj(kerndata) for kerndata in data['kern_list']]
        kernel = self(kern_list, dims=data['dims'])
        return kernel

    def set_cache_state(self, state: bool) -> None:
        self.cache_state = state
        for k in self.kern_list:
            k.set_cache_state(state)


class AdditionKernel(Kernel):
    def __init__(self, kern_list: tp.List[Kernel]) -> None:
        self.name = 'kern_operation.AdditionKernel'
        self.nk = len(kern_list)
        self.kern_list = kern_list
        assert np.all(np.array([k.nout for k in kern_list]) == kern_list[0].nout)
        self.nout = kern_list[0].nout
        self.nps = [len(k.ps) for k in self.kern_list]
        self.cumsum = np.concatenate([np.array([0]), np.cumsum(self.nps)])
        self.ps = concatenate([k.ps for k in self.kern_list])
        self.ps_bound = concatenate([k.ps_bound for k in self.kern_list])
        self.dK_dps = []
        self.transformations = concatenate([k.transformations for k in self.kern_list])
        self.default_cache: tp.Dict[str, tp.Any] = {}
        self.dK_dps = []
        for i in range(len(self.ps)):
            kern_index, pos_index = self.get_pos(i)

            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp(i, X, X2, **kwargs)
            self.dK_dps.append(func)
        super().__init__()
        self.check()

    def get_pos(self, i: int) -> tp.Tuple[int, int]:
        kern_index = len(np.where(self.cumsum <= i)[0]) - 1
        pos_index = i - self.cumsum[kern_index]
        return (kern_index, pos_index)

    @Cache('g')
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        assert np.all(np.array([len(item) for item in X]) == self.nk)
        result = sum(self.kern_list[i].K(self.get_subX(X, i), self.get_subX(X2, i)) for i in range(self.nk))
        return result

    def get_subX(self, X, kern_index):
        xp = utils.get_array_module(X)
        result = [x[kern_index] for x in X]
        try:
            return xp.array(result)
        except:
            return result

    @Cache('g')
    def dK_dp(self, i: int, X, X2=None):
        if X2 is None:
            X2 = X
        assert np.all(np.array([len(item) for item in X]) == self.nk)
        kern_index, pos_index = self.get_pos(i)
        return self.kern_list[kern_index].dK_dps[pos_index](self.get_subX(X, kern_index), self.get_subX(X2, kern_index))

    def clear_cache(self) -> None:
        self.cache_data = {}
        for i in range(self.nk):
            self.kern_list[i].clear_cache()

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        data = {
            'name': self.name,
            'kern_list': [k.to_dict() for k in self.kern_list],
        }
        return data

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> Kernel:
        kern_list = [get_kern_obj(kerndata) for kerndata in data['kern_list']]
        kernel = self(kern_list)
        return kernel

    def set_cache_state(self, state: bool) -> None:
        self.cache_state = state
        for k in self.kern_list:
            k.set_cache_state(state)
