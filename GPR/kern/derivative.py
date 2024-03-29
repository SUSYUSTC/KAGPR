import typing as tp
import numpy as np
from .kernel import Kernel
from .cache import Cache
from . import get_kern_obj
from .param import Param
from . import param_transformation
from .. import utils


class GeneralDerivative(Kernel):
    orders: tp.List[tp.List[int]]
    splits: tp.List[tp.List[int]]
    def __init__(self, kernel: Kernel, n: int, d: int, likelihood_split_type: str) -> None:
        self.default_cache: tp.Any = {}
        self.n = n
        self.d = d
        self.input_dim = (n + 1) * d
        self.dim_K = slice(0, self.d)
        self.dims_grad = [slice(self.d * (i + 1), self.d * (i + 2)) for i in range(self.n)]
        self.kernel = kernel
        self.ps = self.kernel.ps
        self.ps_bound = self.kernel.ps_bound
        self.dK_dps = []
        for i in range(len(self.kernel.ps)):
            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp(i, X, X2, **kwargs)
            self.dK_dps.append(func)

        assert likelihood_split_type in ['same', 'order', 'full']
        self.likelihood_split_type = likelihood_split_type
        if likelihood_split_type == 'same':
            self.splits = [list(range(self.nout))]
        elif likelihood_split_type == 'order':
            self.splits = self.orders
        else:
            self.splits = [[i] for i in range(self.nout)]
        self.n_likelihood_splits = self.kernel.n_likelihood_splits * len(self.splits)
        self.transformations = self.kernel.transformations
        super().__init__()

    def split_likelihood(self, Nin: int) -> tp.List[np.ndarray]:
        likelihood_splits = self.kernel.split_likelihood(Nin)
        results: tp.List[np.ndarray] = []
        for s in likelihood_splits:
            for o in self.splits:
                results.append(np.concatenate([s + Nin * i for i in o]))
        assert len(results) == self.n_likelihood_splits
        return results

    def _fake_K(self, X, X2, K, dK_dX, dK_dX2, d2K_dXdX2):
        raise NotImplementedError

    def K(self, X, X2=None):
        raise NotImplementedError

    def dK_dp(self, i, X, X2=None):
        raise NotImplementedError

    def Kdiag(self, X):
        raise NotImplementedError

    def clear_cache(self) -> None:
        self.cache_data = {}
        self.kernel.clear_cache()

    def set_cache_state(self, state: bool) -> None:
        self.cache_state = state
        self.kernel.set_cache_state(state)


class FullDerivative(GeneralDerivative):
    def __init__(self, kernel: Kernel, n: int, d: int, optfactor: bool=False, likelihood_split_type: str='same') -> None:
        self.name = 'derivative.FullDerivative'
        self.nout = n + 1
        self.orders = [[0], list(range(1, self.nout))]
        super().__init__(kernel, n, d, likelihood_split_type)
        self.factor = Param('factor', 1.0)
        self.optfactor = optfactor
        if optfactor:
            self.ps.append(self.factor)
            self.dK_dps.append(self.dK_dfactor)
            self.ps_bound.append((0, np.inf))
            self.transformations.append(param_transformation.log)
        self.check()

    def set_factor(self, factor: utils.general_float) -> None:
        self.factor.value = float(factor)

    def _fake_K(self, X, X2, K, dK_dX, dK_dX2, d2K_dXdX2, d_factor=False):
        if d_factor:
            c0 = 0
            c1 = 1
            c2 = self.factor.value * 2
        else:
            c0 = 1
            c1 = self.factor.value
            c2 = self.factor.value ** 2

        xp = utils.get_array_module(X)
        if X2 is None:
            X2 = X
        N = len(X)
        N2 = len(X2)
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        X2_K = X2[:, self.dim_K]
        X2_grad = [X2[:, dim] for dim in self.dims_grad]
        result = xp.zeros((N * (self.n + 1), N2 * (self.n + 1)))
        result[0:N, 0:N2] = K(X[:, self.dim_K], X2[:, self.dim_K], cache={'g': 0}) * c0
        for i in range(self.n):
            result[N * (i + 1): N * (i + 2), 0:N2] = dK_dX(X_K, X_grad[i], X2=X2_K, cache={'gd1': i, 'g': 0}) * c1
        for j in range(self.n):
            result[0:N, N2 * (j + 1):N2 * (j + 2)] = dK_dX2(X_K, X2_grad[j], X2=X2_K, cache={'gd2': j, 'g': 0}) * c1
        for i in range(self.n):
            for j in range(self.n):
                result[N * (i + 1): N * (i + 2), N2 * (j + 1):N2 * (j + 2)] = d2K_dXdX2(X_K, X_grad[i], X2_grad[j], X2=X2_K, cache={'gdd': (i, j), 'gd1': i, 'gd2': j, 'g': 0}) * c2
        return result

    @Cache('g')
    def K(self, X, X2=None):
        '''
        X, X2: (N, d*(n+1))
        '''
        return self._fake_K(X, X2, self.kernel.K, self.kernel.dK_dX, self.kernel.dK_dX2, self.kernel.d2K_dXdX2)

    @Cache('g')
    def dK_dp(self, i, X, X2=None):
        return self._fake_K(X, X2, self.kernel.dK_dps[i], self.kernel.d2K_dpsdX[i], self.kernel.d2K_dpsdX2[i], self.kernel.d3K_dpsdXdX2[i])

    @Cache('g')
    def dK_dfactor(self, X, X2=None):
        return self._fake_K(X, X2, self.kernel.K, self.kernel.dK_dX, self.kernel.dK_dX2, self.kernel.d2K_dXdX2, True)

    @Cache('g')
    def Kdiag(self, X):
        xp = utils.get_array_module(X)
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.K_0(X_K)] + [self.kernel.d2K_dXdX_0(dX) for dX in X_grad])

    def dK_dldiag(self, X):
        xp = utils.get_array_module(X)
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.dK_dl_0(X_K)] + [self.kernel.d3K_dldXdX_0(dX) for dX in X_grad])

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        data = {
            'name': self.name,
            'n': self.n,
            'd': self.d,
            'optfactor': self.optfactor,
            'factor': self.factor.value,
            'likelihood_split_type': self.likelihood_split_type,
            'kern': self.kernel.to_dict(),
        }
        return data

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> Kernel:
        n = data['n']
        d = data['d']
        if 'factor' in data:
            factor = data['factor']
        else:
            factor = 1.0
        if 'optfactor' in data:
            optfactor = data['optfactor']
        else:
            optfactor = False
        if 'likelihood_split_type' in data:
            likelihood_split_type = data['likelihood_split_type']
        else:
            likelihood_split_type = 'same'
        kern_dict = data['kern']
        kernel = get_kern_obj(kern_dict)
        result = self(kernel, n, d, optfactor=optfactor, likelihood_split_type=likelihood_split_type)
        result.set_factor(factor)
        return result


class Derivative(GeneralDerivative):
    def __init__(self, kernel: Kernel, n: int, d: int, likelihood_split_type: str = 'same'):
        self.name = 'derivative.Derivative'
        self.nout = n
        self.orders = [list(range(self.nout))]
        super().__init__(kernel, n, d, likelihood_split_type)
        self.check()

    def _fake_K(self, X, X2, d2K_dXdX2):
        xp = utils.get_array_module(X)
        if X2 is None:
            X2 = X
        N = len(X)
        N2 = len(X2)
        X_K = X[:, self.dim_K]
        X_grad = [X[:, dim] for dim in self.dims_grad]
        X2_K = X2[:, self.dim_K]
        X2_grad = [X2[:, dim] for dim in self.dims_grad]
        result = xp.zeros((N * self.n, N2 * self.n))
        for i in range(self.n):
            for j in range(self.n):
                result[N * i: N * (i + 1), N2 * j:N2 * (j + 1)] = d2K_dXdX2(X_K, X_grad[i], X2_grad[j], X2=X2_K, cache={'gdd': (i, j), 'gd1': i, 'gd2': j, 'g': 0})
        return result

    @Cache('g')
    def K(self, X, X2=None):
        '''
        X, X2: (N, d)
        '''
        return self._fake_K(X, X2, self.kernel.d2K_dXdX2)

    @Cache('g')
    def dK_dp(self, i, X, X2=None):
        return self._fake_K(X, X2, self.kernel.d3K_dpsdXdX2[i])

    @Cache('g')
    def Kdiag(self, X):
        xp = utils.get_array_module(X)
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.d2K_dXdX_0(dX) for dX in X_grad])

    def dK_dldiag(self, X):
        xp = utils.get_array_module(X)
        X_grad = [X[:, dim] for dim in self.dims_grad]
        return xp.concatenate([self.kernel.d3K_dldXdX_0(dX) for dX in X_grad])

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        data = {
            'name': self.name,
            'n': self.n,
            'd': self.d,
            'likelihood_split_type': self.likelihood_split_type,
            'kern': self.kernel.to_dict()
        }
        return data

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> Kernel:
        n = data['n']
        d = data['d']
        if 'likelihood_split_type' in data:
            likelihood_split_type = data['likelihood_split_type']
        else:
            likelihood_split_type = 'same'
        kern_dict = data['kern']
        kernel = get_kern_obj(kern_dict)
        result = self(kernel, n, d, likelihood_split_type)
        return result
