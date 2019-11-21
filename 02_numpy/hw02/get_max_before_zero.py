import numpy as np


def get_max_before_zero(x):
    cnt_zeros = x.shape[0] - np.count_nonzero(x)
    if cnt_zeros == 0 or cnt_zeros == 1 and x[-1] == 0:
        return None
    idx_after_zero = np.where(np.concatenate([[1], x]) == 0)[0]
    idx_after_zero[idx_after_zero > x.shape[0] - 1] = idx_after_zero[0]
    return np.max(x[idx_after_zero])
