from collections import defaultdict


def _isclose(a, b):
    return abs(a - b) <= 1e-9


class CooSparseMatrix:
    def _getshape(self):
        return self._shape

    def _setshape(self, new_shape):
        if (
                not isinstance(new_shape[0], int) or not isinstance(new_shape[1], int) or
                self._shape[0] * self._shape[1] != new_shape[0] * new_shape[1]
        ):
            raise TypeError
        self._storage = defaultdict(
            float, {
                (self._convert_indices(self._shape, new_shape, (idx, jdx)), val)
                for (idx, jdx), val in self._storage.items()
            }
        )
        self._shape = new_shape

    shape = property(_getshape, _setshape)

    def _gett(self):
        res = CooSparseMatrix([], (self._shape[1], self._shape[0]))
        res._storage = defaultdict(
            float, {((jdx, idx), val) for (idx, jdx), val in self._storage.items()}
        )
        return res

    def _sett(self, val):
        raise AttributeError

    T = property(_gett, _sett)

    @staticmethod
    def _convert_indices(old_shape, new_shape, indices):
        ravel = indices[0] * old_shape[1] + indices[1]
        return ravel // new_shape[1], ravel % new_shape[1]

    def __init__(self, ijx_list, shape):
        self._storage = defaultdict(float)
        self._shape = shape
        for (idx, jdx, val) in ijx_list:
            if (
                not self._check_indices((idx, jdx)) or
                (idx, jdx) in self._storage
            ):
                raise TypeError
            self._storage[(idx, jdx)] = val

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if self._check_indices(item):
                if item in self._storage:
                    return self._storage[item]
                else:
                    return 0.0
            else:
                raise TypeError
        else:
            if not self._check_indices((item, 0)):
                raise TypeError
            line = CooSparseMatrix([], (1, self._shape[1]))
            line._storage = defaultdict(
                float, {((0, jdx), val) for (idx, jdx), val in self._storage.items() if idx == item}
            )
            return line

    def __setitem__(self, key, value):
        if self._check_indices(key):
            if _isclose(value, 0.):
                if key in self._storage:
                    del self._storage[key]
            else:
                self._storage[key] = value
        else:
            raise TypeError

    def __str__(self):
        return self._storage.__str__()

    def _check_indices(self, indices):
        return ((0 <= indices[0] < self._shape[0]) and (0 <= indices[1] < self._shape[1])
                and isinstance(indices[0], int) and isinstance(indices[1], int))

    def __add__(self, other):
        if self._shape[0] != other._shape[0] or self._shape[1] != other._shape[1]:
            raise TypeError
        res = CooSparseMatrix([], self._shape)
        res._storage = defaultdict(float, self._storage)
        for (idx, jdx), val in other._storage.items():
            res._storage[(idx, jdx)] += val
            if _isclose(res._storage[(idx, jdx)], 0.):
                del res._storage[(idx, jdx)]
        return res

    def __sub__(self, other):
        if self._shape[0] != other._shape[0] or self._shape[1] != other._shape[1]:
            raise TypeError
        res = CooSparseMatrix([], self._shape)
        res._storage = defaultdict(float, self._storage)
        for (idx, jdx), val in other._storage.items():
            res._storage[(idx, jdx)] -= val
            if _isclose(res._storage[(idx, jdx)], 0.):
                del res._storage[(idx, jdx)]
        return res

    def __mul__(self, other):
        res = CooSparseMatrix([], self._shape)
        if not _isclose(other, 0.):
            res._storage = defaultdict(
                float, {((idx, jdx), other * val) for (idx, jdx), val in self._storage.items()}
            )
        return res

    __rmul__ = __mul__
