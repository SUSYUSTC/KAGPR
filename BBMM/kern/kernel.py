from . import param
from . import param_transformation
import typing as tp
import numpy as np
from .. import utils
from .cache import Cache


class Kernel(object):
    name: str
    cache_data: tp.Dict[str, tp.Any]
    default_cache: tp.Dict[str, tp.Any]
    ps: tp.List[param.Param]
    ps_bound: tp.List[tp.Tuple[float, float]]
    set_ps: tp.List[tp.Callable[[utils.general_float], None]]
    dK_dps: tp.List[tp.Callable]
    d2K_dpsdX: tp.List[tp.Callable]
    d2K_dpsdX2: tp.List[tp.Callable]
    d3K_dpsdXdX2: tp.List[tp.Callable]
    nout: int
    transformations: tp.List[param_transformation.Transformation]
    n_likelihood_splits: int

    def __init__(self) -> None:
        self.cache_state: bool = True
        self.cache: tp.Dict[str, tp.Any] = {}
        if not hasattr(self, 'n_likelihood_splits'):
            self.n_likelihood_splits = 1

    def check(self):
        assert hasattr(self, 'default_cache')
        assert hasattr(self, 'ps')
        assert hasattr(self, 'dK_dps')
        assert hasattr(self, 'transformations')
        assert hasattr(self, 'nout')
        self.set_ps = []
        for i in range(len(self.ps)):
            def func(value: utils.general_float, i=i):
                return self.set_p(i, value)
            self.set_ps.append(func)

    def finalize(self):
        self.unique_ps, self.unique_ps_indices = param.group_params(self.ps)
        self.unique_ps_first_index = [item[0] for item in self.unique_ps_indices]
        self.unique_transformations = [self.transformations[i] for i in self.unique_ps_first_index]
        self.unique_ps_bound = [self.ps_bound[i] for i in self.unique_ps_first_index]

        self.set_unique_ps = []
        for i in range(len(self.unique_ps)):
            def func(value: utils.general_float, i=i):
                return self.set_unique_p(i, value)
            self.set_unique_ps.append(func)

        self.dK_dps_split = []
        for i in range(len(self.ps)):
            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp_split(i, X, X2, **kwargs)
            self.dK_dps_split.append(func)

        self.dK_dps_unique = []
        for i in range(len(self.unique_ps)):
            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp_unique(i, X, X2, **kwargs)
            self.dK_dps_unique.append(func)

        self.dK_dps_split_unique = []
        for i in range(len(self.unique_ps)):
            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp_split_unique(i, X, X2, **kwargs)
            self.dK_dps_split_unique.append(func)

    def set_p(self, i: int, value: utils.general_float):
        self.ps[i].value = value

    def set_all_ps(self, params: tp.List[utils.general_float]) -> None:
        assert len(params) == len(self.ps)
        for i in range(len(self.ps)):
            self.set_ps[i](params[i])

    def set_unique_p(self, i: int, value:utils.general_float):
        self.unique_ps[i].value = value

    def set_all_unique_ps(self, params: tp.List[utils.general_float]) -> None:
        assert len(params) == len(self.unique_ps)
        for i in range(len(self.unique_ps)):
            self.set_unique_ps[i](params[i])

    def K(self, X1, X2=None, cache: tp.Dict[str, tp.Any] = {}):
        raise NotImplementedError

    def dK_dp_unique(self, i, X, X2=None, **kwargs):
        indices = self.unique_ps_indices[i]
        return sum([self.dK_dps[index](X, X2, **kwargs) for index in indices])

    def dK_dp_split_unique(self, i, X, X2=None, **kwargs):
        indices = self.unique_ps_indices[i]
        return sum([self.dK_dps_split[index](X, X2, **kwargs) for index in indices])

    def split_likelihood(self, Nin: int) -> tp.List[np.ndarray]:
        return [np.arange(Nin)]

    def clear_cache(self):
        raise NotImplementedError

    def set_cache_state(self, state: bool):
        raise NotImplementedError

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        raise NotImplementedError

    def to_dict_final(self) -> tp.Dict[str, tp.Any]:
        d = self.to_dict()
        d['unique_ps'] = list(map(lambda x: x.value, self.unique_ps))
        d['unique_ps_indices'] = self.unique_ps_indices
        return d

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any]) -> 'Kernel':
        raise NotImplementedError

    @classmethod
    def from_dict_final(self, data:tp.Dict[str, tp.Any]) -> 'Kernel':
        kernel = self.from_dict(data)
        if ('unique_ps' in data) and ('unique_ps_indices' in data):
            unique_ps = data['unique_ps']
            unique_ps_indices = data['unique_ps_indices']
            for i in range(len(unique_ps)):
                indices = unique_ps_indices[i]
                name = kernel.ps[indices[0]].name
                p = param.Param(name, unique_ps[i])
                for index in unique_ps_indices[i]:
                    kernel.ps[index] = p
        return kernel

    def copy(self) -> 'Kernel':
        return self.from_dict(self.to_dict())

    def _fake_K_split(self, method, X, X2=None, onetime_number=5000, save_on_CPU=False, progress=False, file=None):
        # Assume X and X2 is always divided for multiGPU case
        multiGPU = False
        try:
            from ..multiGPU import copyed_array
            if isinstance(X, copyed_array):
                multiGPU = True
        except BaseException:
            pass
        if X2 is None:
            X2 = X
        if multiGPU:
            import cupy as cp
        else:
            assert utils.get_array_module(X) is np
            assert utils.get_array_module(X2) is np
        if save_on_CPU or (not multiGPU):
            xp = np
        else:
            xp = cp
        N1 = len(X)
        N2 = len(X2)
        split1 = utils.make_slices(N1, onetime_number)
        split2 = utils.make_slices(N2, onetime_number)
        s1 = len(split1)
        s2 = len(split2)
        index = 0
        result = [[None for i in range(s2)] for i in range(s1)]
        for i, slic1 in enumerate(split1):
            if progress:
                print('progress:', i, 'in', s1, file=file, flush=True)
            for j, slic2 in enumerate(split2):
                if multiGPU:
                    with cp.cuda.Device(index):
                        values = method(X.data[index][slic1], X2.data[index][slic2])
                        result[i][j] = values
                    index = (index + 1) % X.nGPUs
                else:
                    values = method(X[slic1], X2[slic2])
                result[i][j] = values
            if multiGPU:
                for j in range(s2):
                    if save_on_CPU:
                        result[i][j] = cp.asnumpy(result[i][j])
                    else:
                        with cp.cuda.Device(0):
                            result[i][j] = cp.asarray(result[i][j])
            result[i] = xp.concatenate(result[i], axis=1)
        result = xp.concatenate(result)
        return result

    @Cache('no')
    def K_split(self, X, X2=None, onetime_number=5000, save_on_CPU=False, progress=False, file=None):
        return self._fake_K_split(self.K, X, X2, onetime_number=onetime_number, save_on_CPU=save_on_CPU, progress=progress, file=file)

    @Cache('no')
    def dK_dp_split(self, i, X, X2=None, onetime_number=5000, save_on_CPU=False, progress=False, file=None):
        return self._fake_K_split(self.dK_dps[i], X, X2, onetime_number=onetime_number, save_on_CPU=save_on_CPU, progress=progress, file=file)
