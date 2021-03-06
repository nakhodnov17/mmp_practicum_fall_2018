def linearize(container):
    for obj in container:
        if isinstance(obj, str) and len(obj) == 1:
            yield obj
        else:
            try:
                _ = iter(obj)
                yield from linearize(obj)
            except TypeError:
                yield obj
