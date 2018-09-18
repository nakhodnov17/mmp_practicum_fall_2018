def subreverse_sequence(sequence):
    return tuple(sequence[::2][::-1] + sequence[1::2])
