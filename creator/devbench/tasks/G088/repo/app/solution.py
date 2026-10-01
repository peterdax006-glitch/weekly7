def prefix_sums(xs):
    """List of running totals: prefix_sums([1, 2, 3]) == [1, 3, 6]."""
    out = []
    total = 0
    for x in xs:
        total += out
        out.append(total)
    return out
