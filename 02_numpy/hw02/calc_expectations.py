import numpy as np


def calc_expectations(h, w, X, Q):
    rec_cumsum = np.cumsum(np.cumsum(Q, axis=0), axis=1)
    idx = np.array(range(0, Q.shape[0]))
    jdx = np.array(range(0, Q.shape[1]))

    idx_l_t, jdx_l_t = idx - h, jdx - w
    rec_cumsum_l_t = np.zeros_like(Q)
    rec_cumsum_l_t[np.meshgrid(idx[idx_l_t >= 0],
                               jdx[jdx_l_t >= 0])] = \
        rec_cumsum[np.meshgrid(idx_l_t[idx_l_t >= 0],
                               jdx_l_t[jdx_l_t >= 0])]

    idx_r_t, jdx_r_t = idx - h, jdx
    rec_cumsum_r_t = np.zeros_like(Q)
    rec_cumsum_r_t[np.meshgrid(idx[idx_r_t >= 0],
                               jdx[jdx_r_t >= 0])] = \
        rec_cumsum[np.meshgrid(idx_r_t[idx_r_t >= 0],
                               jdx_r_t[jdx_r_t >= 0])]

    idx_l_b, jdx_l_b = idx, jdx - w
    rec_cumsum_l_b = np.zeros_like(Q)
    rec_cumsum_l_b[np.meshgrid(idx[idx_l_b >= 0],
                               jdx[jdx_l_b >= 0])] = \
        rec_cumsum[np.meshgrid(idx_l_b[idx_l_b >= 0],
                               jdx_l_b[jdx_l_b >= 0])]

    propas = rec_cumsum + rec_cumsum_l_t - rec_cumsum_l_b - rec_cumsum_r_t

    return propas * X
