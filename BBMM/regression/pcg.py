import numpy as np
import numpy.linalg as LA
import scipy.linalg as SLA
import time
from .preconditioner import Preconditioner_Nystroem
from .krylov import Krylov


def PCG(K: np.ndarray, diag_reg, y: np.ndarray, Nk: int, thres: float=1e-6, verbose=True, file=None):
    if verbose:
        print('Start PCG calculation', file=file)
    N = len(y)
    init_indices = np.random.permutation(N)[0:Nk]
    init_indices = np.sort(init_indices)
    if verbose:
        print('Start constructing Nystroem preconditioner', file=file, flush=True)
        print('Total size:', N, file=file, flush=True)
        print('Preconditioner size:', Nk, file=file, flush=True)
    t1 = time.time()
    K11 = np.array(K[init_indices][:, init_indices])
    K21 = np.array(K[:, init_indices])
    diag_reg_11 = diag_reg[init_indices]
    K21 = K21 / np.sqrt(diag_reg[:, None])
    Q, R = np.linalg.qr(K21)
    if verbose:
        print('QR done', file=file, flush=True)
    del K21
    L11 = np.linalg.cholesky(K11 + np.diag(diag_reg_11))
    R = SLA.solve_triangular(L11, R.T, lower=True, trans=0).T
    u, s, vh = np.linalg.svd(R)
    if verbose:
        print('SVD done', file=file, flush=True)
    del K11, R, L11
    U = Q.dot(u)
    Lambda = np.square(s)
    del u, s, vh
    pred_nystroem = Preconditioner_Nystroem(Lambda, U, diag_reg, 0)
    t2 = time.time()
    if verbose:
        print('Preconditioner done. Time:', t2 - t1, file=file, flush=True)

    def mv(w):
        result = w
        result = pred_nystroem.mv_invhalf(result)
        result = K.dot(result) + result * diag_reg[:, None]
        result = pred_nystroem.mv_invhalf(result)
        return result

    def callback(i, residual, t_cg):
        if verbose:
            print(i, residual, file=file)

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
    return w
