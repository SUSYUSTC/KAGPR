import numpy as np
import time
import GPR
import GPy
import unittest

params_ref = np.array([1.558044e+02, 1.000000e+00, 2.421621e-05])
ll_ref = 9.580174e+02

class Test(unittest.TestCase):
    def _run(self, GPU):
        X = np.load("./X_test.npy")
        Y = np.load("./Y_test.npy")
        noise = 1e-5
        k = GPR.kern.RBF()
        lengthscale = 1.0
        variance = 1.0
        k.set_lengthscale(lengthscale)
        k.set_variance(variance)
        begin = time.time()
        gp = GPR.GP(X, Y, k, noise, GPU=GPU)
        gp.optimize(messages=False, active_params=[True, False, True])
        err = np.max(np.abs((gp.params - params_ref) / params_ref))
        self.assertTrue(err < 1e-4)
        err = np.abs((gp.ll - ll_ref) / ll_ref)
        self.assertTrue(err < 1e-6)

    def test_CPU(self):
        self._run(False)

    def test_GPU(self):
        self._run(True)


if __name__ == '__main__':
    unittest.main()
