import numpy as np
import unittest
import GPR

# y = sum_i sin(sum_k x_ik), with gradients along n directions per element
n = 3
d = 4


def func(x):
    X_K = x[:, :d]
    X_grad = [x[:, d * (i + 1):d * (i + 2)] for i in range(n)]
    c = np.cos(np.sum(X_K, axis=1))
    return [np.sum(np.sin(np.sum(X_K, axis=1)))] + [np.sum(c * np.sum(dX, axis=1)) for dX in X_grad]


class Test(unittest.TestCase):
    def test_K(self):
        np.random.seed(0)
        sizes1 = np.random.randint(1, 6, size=(5, ))
        sizes2 = np.random.randint(1, 6, size=(4, ))
        X1 = [np.random.random((s, d * (n + 1))) for s in sizes1]
        X2 = [np.random.random((s, d * (n + 1))) for s in sizes2]
        kernel = GPR.kern.FullDerivative(GPR.kern.Matern52(), n, d, likelihood_split_type='order')
        kernel_summation = GPR.kern.Summation(kernel)
        self.assertTrue(kernel_summation.split_likelihood(len(sizes1))[1].tolist() == list(range(len(sizes1), len(sizes1) * (n + 1))))
        for method, method_summation in zip([kernel.K] + kernel.dK_dps, [kernel_summation.K] + kernel_summation.dK_dps):
            K = method_summation(X1, X2)
            self.assertTrue(K.shape == (len(sizes1) * (n + 1), len(sizes2) * (n + 1)))
            for i1 in range(len(sizes1)):
                for i2 in range(len(sizes2)):
                    kernel_summation.clear_cache()
                    K_full = method(X1[i1], X2[i2])
                    # rows/columns of the summed kernel: output o of molecule i is at o * N + i
                    K_ref = K_full.reshape(n + 1, sizes1[i1], n + 1, sizes2[i2]).sum(axis=(1, 3))
                    err = np.max(np.abs(K[i1::len(sizes1)][:, i2::len(sizes2)] - K_ref))
                    self.assertTrue(err < 1e-10)
            kernel_summation.clear_cache()

    def test_GP(self):
        np.random.seed(0)
        sizes = np.random.randint(2, 6, size=(40, ))
        X_train = [np.random.random((s, d * (n + 1))) for s in sizes]
        Y_train = np.array([func(x) for x in X_train]).T.reshape(-1, 1)
        X_test = [np.random.random((s, d * (n + 1))) for s in np.random.randint(2, 6, size=(10, ))]
        Y_test = np.array([func(x) for x in X_test]).T.reshape(-1, 1)
        kernel = GPR.kern.FullDerivative(GPR.kern.Matern52(), n, d, likelihood_split_type='order')
        kernel_summation = GPR.kern.Summation(kernel)
        gp = GPR.GP(X_train, Y_train, kernel_summation, [1e-4, 1e-4])
        gp.optimize(messages=False)
        pred_test = gp.predict(X_test)
        rel_err = np.sqrt(np.mean((pred_test - Y_test)**2) / np.mean(Y_test**2))
        self.assertTrue(rel_err < 0.05)


if __name__ == '__main__':
    unittest.main()
