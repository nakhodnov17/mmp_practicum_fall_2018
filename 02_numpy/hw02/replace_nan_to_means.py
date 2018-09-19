import numpy as np


def replace_nan_to_means(x):
    x = np.copy(x)
    x[:, np.count_nonzero(np.isnan(x), axis=0) == x.shape[1]] = 0.
    means = np.nanmean(x, axis=0).reshape([1, -1])
    x[np.isnan(x)] = np.repeat(means, x.shape[0], axis=0)[np.isnan(x)]
    return x
