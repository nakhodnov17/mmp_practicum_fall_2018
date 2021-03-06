import numpy as np


def BatchGenerator(list_of_sequences, batch_size, shuffle=False):
    n_batchs = int(
        np.ceil(len(list_of_sequences[0]) / float(batch_size))
    )
    if shuffle:
        idxs = np.random.permutation(np.arange(0, n_batchs))
    else:
        idxs = np.arange(0, n_batchs)
    for idx in idxs:
        left_border = batch_size * idx
        right_border = min(len(list_of_sequences[0]), left_border + batch_size)
        yield [seq[left_border:right_border] for seq in list_of_sequences]
