def WordContextGenerator(words, window_size):
    for idx in range(len(words)):
        for jdx in range(max(0, idx - window_size), min(len(words), idx + window_size + 1)):
            if idx != jdx:
                yield words[idx], words[jdx]
