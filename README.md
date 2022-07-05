# Gaussian Process (GP) and Revised BBMM methods for GP

## Installation ##
```python setup.py install```

## Dependences ##
* cupy ```pip install cupy ``` 

## How to use ##
Create data and kernel
```
import numpy as np
import GPR
X = np.random.random((100, 10))
Y = np.sum(np.sin(X), axis=1)[:, None]
noise = 1e-4
k = GPR.kern.RBF()
k.set_lengthscale(1.0)
k.set_variance(10.0)
```

Train a full GP model and then save
```
gp = GPR.GP(X, Y, k, noise, GPU=False)
gp.optimize(messages=False)
# predict
Y_pred = gp.predict(X)
# save
gp.save("model.npz")
```

Train a BBMM model and then save
```
bbmm = GPR.BBMM(k, nGPU=1)
bbmm.initialize(X, noise)
bbmm.set_preconditioner(50, nGPU=0)
bbmm.solve_iter(Y)
# predict
Y_pred = bbmm.predict(X)
# save
bbmm.save("model.npz")
```

An KA-GPR example to learn $\sum^{N}_{i=0} \sin(\sum_{j=1}^{d} x_{ij})$ with varying N.
Using training data with $5\leq N \leq 10$, we can predict data with $10\leq N \leq 20$.
```
import numpy as np
import GPR


def func(x):
    return np.sum(np.sin(np.sum(x, axis=1)))


d = 5
# 20 test data with N from 10-20
sizes_test = np.random.randint(10, 20, size=(20, ))
X_test = [np.random.random((n, d)) for n in sizes_test]
Y_test = np.array([func(x) for x in X_test])[:, None]
rel_errs = []
for N in [5, 10, 20, 30, 50]:
    # training data with N from 5-10
    sizes = np.random.randint(5, 10, size=(N, ))
    X_train = [np.random.random((n, d)) for n in sizes]
    Y_train = np.array([func(x) for x in X_train])[:, None]
    kernel = GPR.kern.RBF()
    kernel_summation = GPR.kern.Summation(kernel)
    gp = GPR.GP(X_train, Y_train, kernel_summation, 1e-4)
    gp.optimize(messages=False)
    Y_test = np.array([func(x) for x in X_test])[:, None]
    pred_test = gp.predict(X_test)
    rel_errs.append(np.mean(np.abs((pred_test - Y_test) / Y_test)))
print(*rel_errs, sep='\n')
```
The result depends on the machine and random seed. What I got is

0.0997368608827424

0.0722073273267897

0.03642092660762913

0.015870446068140552

0.008450891824084485

The kernel calculations are cached by default. If you want to play with them by yourself you may want `YOUR_KERNEL.clear_cache` or disable the cacheing by `YOUR_KERNEL.set_cache_state(False)`.

## References ##
1. Wang, Ke Alexander, Geoff Pleiss, Jacob R. Gardner, Stephen Tyree, Kilian Q. Weinberger, and Andrew Gordon Wilson. “Exact Gaussian processes on a million data points.” arXiv preprint arXiv:1903.08114 (2019). Accepted by NeurIPS 2019 [[Link]](https://arxiv.org/abs/1903.08114)
2. Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. (2018). Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. arXiv preprint arXiv:1809.11165. Accepted by NeurIPS 2018 [[Link]](https://arxiv.org/abs/1809.11165)
3. Sun, J., Cheng, L., & Miller III, T. F. (2021). Molecular Energy Learning Using Alternative Blackbox Matrix-Matrix Multiplication Algorithm for Exact Gaussian Process. arXiv preprint arXiv:2109.09817. [[Link]](https://arxiv.org/abs/2109.09817)
4. Sun, J., Cheng, L., & Miller III, T. F. (2022). Molecular Dipole Moment Learning via Rotationally Equivariant Gaussian Process Regression with Derivatives in Molecular-orbital-based Machine Learning. arXiv:2205.15510. [[Link]](https://arxiv.org/pdf/2205.15510)
