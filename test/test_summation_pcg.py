import numpy as np
import unittest
import BBMM
import cupy as cp
import os


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
        kernel = BBMM.kern.RBF()
        kernel_summation = BBMM.kern.Summation(kernel)
        kernel_summation.set_all_ps(params_ref[0:-1])
        gp = BBMM.GP(X_train, Y_train, kernel_summation, params_ref[-1], GPU=True)
        kernel_summation.set_save_on_CPU(True)
        assert isinstance(gp.X[0], cp.ndarray)
        K = kernel_summation.K(gp.X)
        kernel_summation.set_save_on_CPU(False)
        assert isinstance(K, np.ndarray)
        w = BBMM.PCG(K, gp.diag_reg, Y_train, 3, thres=1e-8, verbose=False)
        assert isinstance(w, np.ndarray)
        gp.input_w(cp.asarray(w))
        pred_train = gp.predict(X_train)
        err = np.max(np.abs(pred_train - Y_train))
        self.assertTrue(err < 2e-4)
        pred_test = gp.predict(X_test)
        rel_err = np.max(np.abs((pred_test - Y_test) / Y_test))
        self.assertTrue(rel_err < 0.1)
        gp.save("model")
        err = np.max(np.abs(BBMM.GP.load("model.npz", True).predict(X_test) - pred_test))
        self.assertTrue(err < 1e-10)
        os.remove("model.npz")


if __name__ == '__main__':
    unittest.main()
