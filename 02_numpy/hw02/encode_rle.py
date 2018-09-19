import numpy as np


def encode_rle(x):
    x_left_nan = np.concatenate([[np.nan], x])
    x_right_nan = np.concatenate([x, [np.nan]])

    idx_unique = ~(x_right_nan == np.roll(x_right_nan, -1))[:-1]
    arr_1 = x[idx_unique]

    right_idx = np.argwhere(~(x_right_nan == np.roll(x_right_nan, -1))[:-1]).reshape(-1)
    left_idx = np.argwhere(~(x_left_nan == np.roll(x_left_nan, 1))[1:]).reshape(-1)
    arr_2 = right_idx - left_idx + 1

    return arr_1, arr_2
