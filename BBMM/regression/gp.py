import typing as tp
from collections.abc import Iterable
import numpy as np
from ..kern import Kernel, get_kern_obj, param_transformation
import time
import sys
from .noise import Noise
from .. import utils
from .pcg import PCG


class GP(object):
    def __init__(self, X: np.ndarray, Y: np.ndarray, kernel: Kernel, noise: tp.Union[utils.general_float, tp.Iterable[utils.general_float]], GPU: tp.Union[bool, int] = False, split: bool = False, file=None):
        self.Nin = len(X)
        self.kernel = kernel
        self.kernel.finalize()
        self.Nout = self.Nin * self.kernel.nout
        assert len(Y) == self.Nout
        self.noise = Noise(noise, self.kernel.n_likelihood_splits)
        self.likelihood_splits = self.kernel.split_likelihood(self.Nin)
        self.diag_reg = self.noise.get_diag_reg(self.likelihood_splits)
        self.diag_reg_gradient = self.noise.get_diag_reg_gradient(self.likelihood_splits)
        self.nks = len(self.kernel.ps)
        self.unique_nks = len(self.kernel.unique_ps)
        self.nns = len(self.noise.values)
        self.transformations_group = param_transformation.Group(kernel.transformations + [param_transformation.log] * self.nns)
        self.unique_transformations_group = param_transformation.Group(kernel.unique_transformations + [param_transformation.log] * self.nns)
        self.split = split
        if isinstance(GPU, bool):
            self.GPU = GPU
            self.nGPUs = int(GPU)
        else:
            self.GPU = GPU > 0
            self.nGPUs = GPU
        self.kernel_options = {}
        if self.GPU:
            import cupy as cp
            self.xp = cp
            import cupyx
            import cupyx.scipy.linalg
            from ..multiGPU import copyed_array
            cupyx.seterr(linalg='raise')
            self.xp_solve_triangular = cupyx.scipy.linalg.solve_triangular
            if split:
                self.X = copyed_array(X, self.nGPUs)
            else:
                self.X = utils.apply_recursively(cp.asarray, X)
            self.Y = cp.asarray(Y).copy()
        else:
            import scipy
            self.xp = np
            self.xp_solve_triangular = scipy.linalg.solve_triangular
            self.X = X
            self.Y = Y
        if split:
            self.kernel_K = self.kernel.K_split
            self.kernel_dK_dps = self.kernel.dK_dps_split_unique
        else:
            self.kernel_K = self.kernel.K
            self.kernel_dK_dps = self.kernel.dK_dps_unique
        if file is None:
            self.file = sys.__stdout__
        else:
            self.file = file

    def set_kernel_options(self, **options):
        self.kernel_options = options

    def input_w(self, w) -> None:
        self.w = w
        self.grad = False

    def fit(self, grad: bool = False) -> None:
        self.grad = grad
        self.diag_reg = self.noise.get_diag_reg(self.likelihood_splits)
        K_noise = self.kernel_K(self.X, cache=self.kernel.default_cache, **self.kernel_options)
        K_noise[self.xp.arange(self.Nout), self.xp.arange(self.Nout)] += self.xp.array(self.diag_reg)
        L = self.xp.linalg.cholesky(K_noise)
        del K_noise
        w_int = self.xp_solve_triangular(L, self.Y, lower=True, trans=0)
        self.w = self.xp_solve_triangular(L, w_int, lower=True, trans=1)
        if grad:
            self.diag_reg_gradient = self.noise.get_diag_reg_gradient(self.likelihood_splits)
            logdet = self.xp.sum(self.xp.log(self.xp.diag(L))) * 2
            Linv = self.xp_solve_triangular(L, self.xp.eye(self.Nout), lower=True, trans=0)
            del L
            K_noise_inv = Linv.T.dot(Linv)
            del Linv
            self.ll = - (self.Y.T.dot(self.w)[0, 0] + logdet) / 2 - self.xp.log(self.xp.pi * 2) * self.Nout / 2
            dL_dK = (self.w.dot(self.w.T) - K_noise_inv) / 2
            del K_noise_inv
            dL_dps = [self.xp.sum(dL_dK * dK_dp(self.X, cache=self.kernel.default_cache, **self.kernel_options)) for dK_dp in self.kernel_dK_dps]
            dL_dns = [self.xp.trace(dL_dK * self.xp.diag(dK_dn_diag)) for dK_dn_diag in self.diag_reg_gradient]
            self.gradient = self.xp.array(dL_dps + dL_dns)
            if self.GPU:
                self.ll = self.ll.get()
                self.gradient = self.gradient.get()

    def save(self, path: str) -> None:
        if self.GPU:
            if self.split:
                X = self.X.data[0]
            else:
                X = self.X
            data = {
                'kernel': self.kernel.to_dict_final(),
                'X': utils.apply_recursively(self.xp.asnumpy, X),
                'Y': self.xp.asnumpy(self.Y),
                'w': self.xp.asnumpy(self.w),
                'noise': self.noise.values,
                'grad': self.grad,
            }
        else:
            data = {
                'kernel': self.kernel.to_dict_final(),
                'X': self.X,
                'Y': self.Y,
                'w': self.w,
                'noise': self.noise.values,
                'grad': self.grad,
            }
        if self.grad:
            data['ll'] = self.ll
            data['gradient'] = self.gradient
        np.savez(path, **data)

    @classmethod
    def from_dict(self, data: tp.Dict[str, tp.Any], GPU: tp.Union[bool, int] = False, split: bool = False) -> 'GP':
        kernel_dict = data['kernel'][()]
        kernel = get_kern_obj(kernel_dict, final=True)
        result = self(data['X'], data['Y'], kernel, noise=data['noise'][()], GPU=GPU, split=split)
        if GPU:
            result.w = result.xp.asarray(data['w'])
        else:
            result.w = data['w']
        return result

    @classmethod
    def load(self, path: str, GPU: tp.Union[bool, int] = False, split: bool = False) -> 'GP':
        data = dict(np.load(path, allow_pickle=True))
        return self.from_dict(data, GPU=GPU, split=split)

    def predict(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self.kernel.clear_cache()
        if self.GPU:
            if self.split:
                from ..multiGPU import copyed_array
                X_GPU = copyed_array(X, self.nGPUs)
            else:
                X_GPU = utils.apply_recursively(self.xp.array, X)
            result = self.xp.asnumpy(self.kernel_K(X_GPU, self.X).dot(self.w))
            if training:
                result += self.xp.asnumpy(self.w) * self.noise.get_diag_reg(self.likelihood_splits)[:, None]
            return result
        else:
            result = self.kernel_K(X, self.X).dot(self.w)
            if training:
                result += self.w * self.noise.get_diag_reg(self.likelihood_splits)[:, None]
            return result

    def update(self, unique_ps: tp.List[utils.general_float], noises: tp.List[float]) -> None:
        self.kernel.set_all_unique_ps(unique_ps)
        self.noise.values = noises
        self.kernel.clear_cache()
        self.params = np.array(unique_ps + noises)

    def objective(self, unique_transform_ps_noise: np.ndarray) -> tp.Tuple[float, np.ndarray]:
        unique_ps_noise = self.unique_transformations_group.inv(unique_transform_ps_noise.tolist())
        if self.messages:
            print('x:' + ' %e' * len(unique_ps_noise) % tuple(unique_ps_noise), file=self.file, flush=True)
        unique_d_transform_ps_noise = self.unique_transformations_group.d(unique_ps_noise)
        self.update(unique_ps_noise[:self.unique_nks], unique_ps_noise[-self.nns:])
        self.fit(grad=True)
        self.unique_transform_gradient = self.gradient / np.array(unique_d_transform_ps_noise)
        result = (-self.ll, -self.unique_transform_gradient)
        return result

    def active_objective(self, params: np.ndarray) -> tp.Tuple[float, np.ndarray]:
        full_params = self.init_params.copy()
        full_params[self.active_params] = params
        ll, gradient = self.objective(full_params)
        return (ll, gradient[self.active_params])

    def get_numerical_gradient(self, transform_ps_noise: np.ndarray, delta: float) -> np.ndarray:
        n_grad = []
        for i in range(len(transform_ps_noise)):
            transform_ps_noise_p = transform_ps_noise.copy()
            transform_ps_noise_p[i] += delta
            obj_p, _ = self.objective(transform_ps_noise_p)
            transform_ps_noise_n = transform_ps_noise.copy()
            transform_ps_noise_n[i] -= delta
            obj_n, _ = self.objective(transform_ps_noise_n)
            n_grad.append((obj_p - obj_n) / (delta * 2))
        return np.array(n_grad)

    def opt_callback(self, x):
        if self.messages:
            print('gradient', self.unique_transform_gradient)
            print('-ll', np.format_float_scientific(-self.ll, precision=6), 'gradient', np.linalg.norm(self.unique_transform_gradient), file=self.file, flush=True)
            print(file=self.file, flush=True)
        self.saved_ps = list(map(lambda p: p.value, self.kernel.ps))
        self.saved_noises = self.noise.values
        update = False
        if not hasattr(self, 'current_best_ll'):
            update = True
        else:
            if self.ll > self.current_best_ll:
                update = True
        if update:
            self.current_best_ll = self.ll
            self.current_best_ps = self.saved_ps
            self.current_best_noises = self.saved_noises

    def optimize(self, messages=False, tol=1e-6, noise_bound: tp.Union[utils.general_float, tp.List[utils.general_float]] = 1e-10, active_params=None) -> None:
        import scipy
        import scipy.optimize
        self.messages = messages
        callback: tp.Callable = self.opt_callback
        begin = time.time()
        noise_bound_list = utils.make_desired_size(noise_bound, self.kernel.n_likelihood_splits)
        unique_ps_noise = list(map(lambda p: p.value, self.kernel.unique_ps)) + self.noise.values
        self.init_params = np.array(self.unique_transformations_group(unique_ps_noise))
        if active_params is None:
            active_params = np.full((len(unique_ps_noise), ), True)
        self.active_params = np.array(active_params)
        import warnings
        n_try = 1
        while True:
            try:
                unique_ps_noise = list(map(lambda p: p.value, self.kernel.unique_ps)) + self.noise.values
                unique_transform_ps_noise = np.array(self.unique_transformations_group(unique_ps_noise))
                bounds: tp.List[tp.Tuple[float, float]] = [(-np.inf, np.inf) for i in range(self.unique_nks)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for b in noise_bound_list:
                        bounds.append((float(np.log(b)), np.inf))
                bounds = np.array(bounds)[active_params].tolist()

                self.result = scipy.optimize.minimize(self.active_objective, unique_transform_ps_noise[self.active_params], jac=True, method='L-BFGS-B', callback=callback, tol=tol, bounds=bounds, options={'maxls': 100})
                if self.result.success or (n_try >= 12):
                    break
                print("Optimization not successful, restarting", file=self.file, flush=True)
                print("current x", self.result.x, file=self.file, flush=True)
                print("current grad", self.result.jac, file=self.file, flush=True)
                print("current p", -self.result.hess_inv.dot(self.result.jac), file=self.file, flush=True)
                n_try += 1
            except np.linalg.LinAlgError:
                noise_bound_list = [item * 2 for item in noise_bound_list]
                print('Cholesky decomposition failed. Try to use a higher noise bound', noise_bound_list, file=self.file, flush=True)
                self.update(self.saved_ps, self.saved_noises)

        self.success = self.result.success
        if self.ll < self.current_best_ll:
            self.update(self.current_best_ps, self.current_best_noises)
            self.fit(grad=True)
            print('Optimization failed, taking the best history value, -ll =', -self.ll, file=self.file, flush=True)
        end = time.time()
        print('time', end - begin, file=self.file, flush=True)
