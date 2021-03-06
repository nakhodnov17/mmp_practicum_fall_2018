import numpy as np


def grad_finite_diff(func, x, y, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    result = np.empty_like(w, dtype=np.float)
    for idx in range(w.shape[0]):
        delta_w = np.zeros_like(w, dtype=np.float)
        delta_w[idx] += eps
        result[idx] = (func(x, y, w + delta_w) - func(x, y, w)) / eps
    return result
