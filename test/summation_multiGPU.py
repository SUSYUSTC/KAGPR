import numpy as np
import cupy as cp
import unittest
import BBMM
import os

nGPUs = 2


def func(x):
    return np.sum(np.sin(np.sum(x, axis=1)))


params_ref = np.array([3.47555621e-01, 9.76080792e-01, 1.00000550e-04])


class Test(unittest.TestCase):
    def test(self):
        np.random.seed(0)
        sizes = np.arange(1, 11) * 10
        X_train = [np.random.random((n, 10)) for n in sizes]
        X_test = [np.random.random((n, 10)) for n in sizes]
        Y_train = np.array([func(x) for x in X_train])[:, None]
        Y_test = np.array([func(x) for x in X_test])[:, None]
        kernel = BBMM.kern.RBF()
        kernel_summation = BBMM.kern.Summation(kernel)
        gp_single = BBMM.GP(X_train, Y_train, kernel_summation, 1e-4, GPU=False)
        gp = BBMM.GP(X_train, Y_train, kernel_summation, 1e-4, GPU=nGPUs, split=True)
        gp.set_kernel_options(onetime_number=2)
        err = np.max(np.abs(cp.asnumpy(gp.kernel_K(gp.X)) - gp_single.kernel_K(gp_single.X)))
        self.assertTrue(err < 1e-10)
        gp.optimize(messages=False)
        self.assertTrue(np.max(np.abs(gp.params / params_ref - 1)) < 1e-4)
        pred_train = gp.predict(X_train)
        err = np.max(np.abs(pred_train - Y_train))
        self.assertTrue(err < 2e-4)
        Y_test = np.array([func(x) for x in X_test])[:, None]
        pred_test = gp.predict(X_test)
        rel_err = np.max(np.abs((pred_test - Y_test) / Y_test))
        self.assertTrue(rel_err < 0.1)
        gp.save("model")
        err = np.max(np.abs(BBMM.GP.load("model.npz", False).predict(X_test) - pred_test))
        self.assertTrue(err < 1e-10)
        os.remove("model.npz")


if __name__ == '__main__':
    unittest.main()
