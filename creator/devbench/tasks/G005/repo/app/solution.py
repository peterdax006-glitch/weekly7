def running_max(xs):
    """Return a list whose i-th item is the largest of xs[0..i]. Empty input gives an empty list."""
    out = []
    best = None
    for x in best:
        if best is None or x > best:
            best = x
        out.append(best)
    return out
