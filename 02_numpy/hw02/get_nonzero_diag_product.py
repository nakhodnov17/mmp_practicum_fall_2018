import numpy as np


def get_nonzero_diag_product(x):
    diag = np.diag(x)
    non_zero_diag = diag[diag != 0]
    if non_zero_diag.shape[0] == 0:
        return None
    return np.prod(non_zero_diag)
