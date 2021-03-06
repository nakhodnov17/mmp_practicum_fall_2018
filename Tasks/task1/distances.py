import numpy as np


def _row_sq_norm(x):
    return np.sum(x ** 2, axis=1)


def euclidean_distance(x, y):
    return np.sqrt(
        _row_sq_norm(x)[:, np.newaxis] +
        _row_sq_norm(y) +
        -2. * np.dot(x, y.T)
    )


def cosine_distance(x, y, epsilon=1e-5):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    distance = (
            np.dot(x, y.T) /
            (np.sqrt(_row_sq_norm(x)[:, np.newaxis]) + epsilon) /
            (np.sqrt(_row_sq_norm(y)[np.newaxis, :]) + epsilon)
    )
    distance *= -1.
    distance += 1.
    return distance
