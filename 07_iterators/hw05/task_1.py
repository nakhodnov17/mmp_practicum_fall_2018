import numpy as np


class RleSequence:
    def __init__(self, input_sequence):
        self.values, self.lengths = self.encode_rle(input_sequence)
        self.ptr, self.bias = 0, 0
        self.len = input_sequence.shape[0]

    def __contains__(self, item):
        return item in self.values

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr == self.values.shape[0]:
            self.ptr = 0
            self.bias = 0
            raise StopIteration
        result = self.values[self.ptr]
        if self.bias == self.lengths[self.ptr] - 1:
            self.ptr += 1
            self.bias = 0
        else:
            self.bias += 1
        return result

    def __getitem__(self, item):
        if isinstance(item, int):
            start = self.len + item if item < 0 else item
            stop = start + 1
            step = 1
        elif isinstance(item, slice):
            start = 0 if item.start is None else item.start
            start = self.len + start if start < 0 else start

            stop = self.len if item.stop is None else min(item.stop, self.len)
            stop = self.len + stop if stop < 0 else stop

            step = 1 if item.step is None else item.step
        else:
            raise NotImplementedError

        idxs = range(start, stop, step)
        result = np.empty([len(idxs)], dtype=self.values.dtype)
        ptr = 0
        idx = 0
        for val in self:
            if idx in idxs:
                result[ptr] = val
                ptr += 1
            idx += 1
        if isinstance(item, int):
            return result[0]
        return result

    @staticmethod
    def encode_rle(x):
        x_left_nan = np.concatenate([[np.nan], x])
        x_right_nan = np.concatenate([x, [np.nan]])

        idx_unique = ~(x_right_nan == np.roll(x_right_nan, -1))[:-1]
        arr_1 = x[idx_unique]

        right_idx = np.argwhere(~(x_right_nan == np.roll(x_right_nan, -1))[:-1]).reshape(-1)
        left_idx = np.argwhere(~(x_left_nan == np.roll(x_left_nan, 1))[1:]).reshape(-1)
        arr_2 = right_idx - left_idx + 1

        return arr_1, arr_2
