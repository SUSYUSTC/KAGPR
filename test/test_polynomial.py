import numpy as np
import time
import BBMM
import GPy
import unittest


class Test(unittest.TestCase):
    def _run(self, GPU):
        np.random.seed(0)
        nX = 50
        nd = 5
        X = np.random.random((nX, nd))
        X2 = np.random.random((nX, nd))
        C = np.random.random((nd, 1))
        Y = (X**np.pi).dot(C)
        noise = 1e-5
        k = BBMM.kern.Polynomial(2, opt=True)
        kp = BBMM.kern.Polynomial(1.999)
        kn = BBMM.kern.Polynomial(2.001)
        dK_dorder_ref = (kp.K(X, X2) - kn.K(X, X2)) / 0.002
        dK_dorder = k.dK_dorder(X, X2)
        err = np.max(np.abs(dK_dorder - dK_dorder_ref))
        gp = BBMM.GP(X, Y, k, noise, GPU=GPU)
        gp.optimize(messages=False)
        err = gp.params[0] - np.pi
        self.assertTrue(err < 1e-6)

    def test_CPU(self):
        self._run(False)

    def test_GPU(self):
        self._run(True)


if __name__ == '__main__':
    unittest.main()
