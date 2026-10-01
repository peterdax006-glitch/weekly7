def flatten(xs):
    """Flatten arbitrarily nested lists into one flat list."""
    out = []
    for x in flatten:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
