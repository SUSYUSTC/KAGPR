import numpy as np
import time
from .preconditioner import Preconditioner_Nystroem
from .krylov import Krylov


def PCG(K, diag_reg, y, Nk, nGPUs=0, thres=1e-6, verbose=True, file=None):
    if verbose:
        print('Start PCG calculation', file=file)
    N = len(y)
    if nGPUs > 0:
        import cupy as cp
        import cupyx.scipy.linalg
        xp = cp
        diag_reg = cp.asarray(diag_reg)
        SLA = cupyx.scipy.linalg
    else:
        import scipy
        xp = np
        SLA = scipy.linalg

    if verbose:
        print('Start constructing Nystroem preconditioner', file=file, flush=True)
        print('Number of GPUs:', nGPUs, file=file, flush=True)
        print('Total size:', N, file=file, flush=True)
        print('Preconditioner size:', Nk, file=file, flush=True)

    t1 = time.time()
    init_indices = np.random.permutation(N)[0:Nk]
    init_indices = np.sort(init_indices)
    diag_reg_11 = diag_reg[init_indices]
    K11 = xp.array(K[init_indices][:, init_indices])
    K21 = xp.array(K[:, init_indices])
    K21 = K21 / xp.sqrt(diag_reg[:, None])
    Q, R = xp.linalg.qr(K21)
    if verbose:
        print('QR done', file=file, flush=True)
    del K21
    L11 = np.linalg.cholesky(K11 + xp.diag(diag_reg_11))
    R = SLA.solve_triangular(L11, R.T, lower=True, trans=0).T
    u, s, vh = xp.linalg.svd(R)
    if verbose:
        print('SVD done', file=file, flush=True)
    del K11, R, L11
    U = Q.dot(u)
    Lambda = xp.square(s)
    del u, s, vh
    pred_nystroem = Preconditioner_Nystroem(Lambda, U, diag_reg, nGPUs)
    t2 = time.time()
    if verbose:
        print('Preconditioner done. Time:', t2 - t1, file=file, flush=True)

    if nGPUs > 0:
        device_split = np.array_split(np.arange(N), nGPUs)
        Ks_GPU = [None for d in device_split]
        for i, d in enumerate(device_split):
            with cp.cuda.Device(i):
                Ks_GPU[i] = cp.asarray(K[d])

    def mv_K_GPU(w):
        result = [None for d in device_split]
        for i, d in enumerate(device_split):
            with cp.cuda.Device(i):
                result[i] = Ks_GPU[i].dot(cp.asarray(w))
        for i, d in enumerate(device_split):
            with cp.cuda.Device(0):
                result[i] = cp.asarray(result[i])
        with cp.cuda.Device(0):
            result = cp.concatenate(result)
        return result

    def mv_K_CPU(w):
        return K.dot(w)

    def mv_K(w, GPU):
        if GPU:
            return mv_K_GPU(w)
        else:
            return mv_K_CPU(w)

    def mv_Knoise(w, GPU):
        return mv_K(w, GPU) + w * diag_reg[:, None]

    def mv(w):
        result = w
        result = pred_nystroem.mv_invhalf(result)
        result = mv_Knoise(result, nGPUs > 0)
        result = pred_nystroem.mv_invhalf(result)
        return result

    def callback(i, residual, t_cg):
        if verbose:
            print(i, residual)

    if nGPUs > 0:
        y = cp.asarray(y)
    y_transform = pred_nystroem.mv_invhalf(y)
    if verbose:
        print('Starting PCG iteration', file=file, flush=True)
    t1 = time.time()
    krylov = Krylov(mv, y_transform, callback=callback, thres=thres)
    w_transform = krylov.run()
    t2 = time.time()
    if verbose:
        print('PCG iterations done. Time:', t2 - t1, file=file, flush=True)
    w = pred_nystroem.mv_invhalf(w_transform)
    if nGPUs > 0:
        w = cp.asnumpy(w)
    return w
