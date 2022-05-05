import numpy as np
import unittest
import BBMM
import os


def func(x):
    return np.sum(np.sin(np.sum(x, axis=1)))


def func_addition(Xs):
    return np.sum([func(X) for X in Xs])


params_ref = np.array([1.39523070e-04, 6.40017590e+04, 1.11795984e+00, 2.25042620e+00, 9.27394284e-05])


class Test(unittest.TestCase):
    def _run(self, GPU):
        np.random.seed(0)
        N_mol = 10
        s1 = 3
        s2 = 8
        d1 = 4
        d2 = 5
        sizes1 = np.arange(1, N_mol+1) * s1
        sizes2 = np.arange(1, N_mol+1) * s2
        X1_train = [np.random.random((n, d1)) for n in sizes1]
        X2_train = [np.random.random((n, d2)) for n in sizes2]
        X1_test = [np.random.random((n, d1)) for n in sizes1]
        X2_test = [np.random.random((n, d2)) for n in sizes2]
        X_train = list(zip(X1_train, X2_train))
        X_test = list(zip(X1_test, X2_test))
        Y_train = np.array([func_addition(x) for x in X_train])[:, None]
        Y_test = np.array([func_addition(x) for x in X_test])[:, None]
        kernel1 = BBMM.kern.RBF()
        kernel2 = BBMM.kern.Matern52()
        kernel1_summation = BBMM.kern.Summation(kernel1)
        kernel2_summation = BBMM.kern.Summation(kernel2)
        kernel_addition_summation = BBMM.kern.AdditionKernel([kernel1_summation, kernel2_summation])
        gp = BBMM.GP(X_train, Y_train, kernel_addition_summation, 1e-4, GPU=GPU)
        gp.optimize(messages=False)
        err = np.max(np.abs(gp.params / params_ref - 1))
        self.assertTrue(err < 1e-4)
        pred_train = gp.predict(X_train)
        err = np.max(np.abs(pred_train - Y_train))
        self.assertTrue(err < 1e-3)
        pred_test = gp.predict(X_test)
        rel_err = np.max(np.abs((pred_test - Y_test) / Y_test))
        self.assertTrue(rel_err < 0.1)
        gp.save("model")
        err = np.max(np.abs(BBMM.GP.load("model.npz", GPU).predict(X_test) - pred_test))
        self.assertTrue(err < 1e-10)
        os.remove("model.npz")

    def test_CPU(self):
        self._run(False)

    def test_GPU(self):
        self._run(True)


if __name__ == '__main__':
    unittest.main()
