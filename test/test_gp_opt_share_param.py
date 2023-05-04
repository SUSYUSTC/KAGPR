import numpy as np
import GPR
import time
import GPy
import unittest
import os


params_ref_1 = np.array([3.80899972e+01, 7.11314155e-01, 1.99112576e-05])
params_ref_2 = params_ref_1.copy()
params_ref_2[0] /= 2


class Test(unittest.TestCase):
    def _run(self, GPU):
        X = np.load("./X_test.npy")
        Y = np.load("./Y_test.npy")
        X1 = X
        X2 = np.repeat(X[:, None, :], 2, axis=1)
        noise = 1e-5
        k1 = GPR.kern.RBF()
        k_temp = GPR.kern.RBF()
        k2 = GPR.kern.AdditionKernel([k_temp, k_temp])
        gp1 = GPR.GP(X1, Y, k1, noise, GPU=GPU)
        gp1.optimize(messages=False)
        gp2 = GPR.GP(X2, Y, k2, noise, GPU=GPU)
        gp2.optimize(messages=False)
        err1 = np.max(np.abs((gp1.params - params_ref_1) / params_ref_1))
        err2 = np.max(np.abs((gp2.params - params_ref_2) / params_ref_2))
        self.assertTrue(err1 < 1e-3)
        self.assertTrue(err2 < 1e-3)
        gp1.save("model1.npz")
        gp2.save("model2.npz")
        gp1_load = GPR.GP.load("./model1.npz", GPU=GPU)
        gp1_load.optimize(messages=False)
        gp2_load = GPR.GP.load("./model2.npz", GPU=GPU)
        gp2_load.optimize(messages=False)
        err1 = np.max(np.abs(gp1_load.predict(X1) - gp1.predict(X1)))
        err2 = np.max(np.abs(gp2_load.predict(X2) - gp2.predict(X2)))
        self.assertTrue(err1 < 1e-4)
        self.assertTrue(err2 < 1e-4)
        os.remove("model1.npz")
        os.remove("model2.npz")

    def test_CPU(self):
        self._run(False)

    def test_GPU(self):
        self._run(True)


if __name__ == '__main__':
    unittest.main()
