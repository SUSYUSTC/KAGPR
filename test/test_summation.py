import numpy as np
import unittest
import BBMM
import os


def func(x):
    return np.sum(np.sin(np.sum(x, axis=1)))


class Test(unittest.TestCase):
    def run(self, GPU):
        np.random.seed(0)
        sizes = np.arange(1, 11) * 10
        N = len(sizes)
        cumsum = np.concatenate([[0], np.cumsum(sizes)])
        X_train = [np.random.random((n, 10)) for n in sizes]
        X_test = [np.random.random((n, 10)) for n in sizes]
        Y_train = np.array([func(x) for x in X_train])[:, None]
        Y_test = np.array([func(x) for x in X_test])[:, None]
        kernel = BBMM.kern.RBF()
        kernel_summation = BBMM.kern.Summation(kernel)
        kernel_summation.set_onetime_number(70)
        K_full = kernel.K(np.concatenate(X_train), np.concatenate(X_test))
        K = kernel_summation.K(X_train, X_test)
        self.assertTrue(K.shape == (len(sizes), len(sizes)))
        for i in range(N):
            for j in range(N):
                err = np.abs(K[i, j] - np.sum(K_full[cumsum[i]:cumsum[i + 1], cumsum[j]: cumsum[j + 1]]))
                self.assertTrue(err < 1e-10)
        gp = BBMM.GP(X_train, Y_train, kernel_summation, 1e-4, GPU=GPU)
        gp.optimize(messages=False)
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

    def test_CPU(self):
        self.run(False)

    def test_GPU(self):
        self.run(True)


if __name__ == '__main__':
    unittest.main()
