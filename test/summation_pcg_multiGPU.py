import numpy as np
import GPR
import unittest
import cupy as cp
import os

nGPUs = 2


def func(x):
    return np.sum(np.sin(np.sum(x, axis=1)))


params_ref = np.array([3.47555621e-01, 9.76080792e-01, 1.00000550e-04])


class Test(unittest.TestCase):
    def test(self):
        np.random.seed(0)
        sizes = np.arange(1, 11) * 10
        N = len(sizes)
        X_train = [np.random.random((n, 10)) for n in sizes]
        X_test = [np.random.random((n, 10)) for n in sizes]
        Y_train = np.array([func(x) for x in X_train])[:, None]
        Y_test = np.array([func(x) for x in X_test])[:, None]
        kernel = GPR.kern.RBF()
        kernel_summation = GPR.kern.Summation(kernel)
        kernel_summation.set_all_ps(params_ref[0:-1])
        gp = GPR.GP(X_train, Y_train, kernel_summation, params_ref[-1], GPU=nGPUs, split=True)
        gp.set_kernel_options(onetime_number=2)
        K = kernel_summation.K_split(gp.X, save_on_CPU=True, onetime_number=3)
        assert isinstance(K, np.ndarray)
        w = GPR.PCG(K, gp.diag_reg, Y_train, 3, nGPUs=nGPUs, thres=1e-8, verbose=False)
        assert isinstance(w, np.ndarray)
        gp.input_w(cp.asarray(w))
        pred_train = gp.predict(X_train)
        err = np.max(np.abs(pred_train - Y_train))
        self.assertTrue(err < 2e-4)
        pred_train = gp.predict(X_train, training=True)
        err = np.max(np.abs(pred_train - Y_train))
        self.assertTrue(err < 1e-7)
        pred_test = gp.predict(X_test)
        rel_err = np.max(np.abs((pred_test - Y_test) / Y_test))
        self.assertTrue(rel_err < 0.1)
        gp.save("model")
        err = np.max(np.abs(GPR.GP.load("model.npz", True).predict(X_test) - pred_test))
        self.assertTrue(err < 1e-10)
        os.remove("model.npz")


if __name__ == '__main__':
    unittest.main()
