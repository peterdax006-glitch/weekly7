def second_largest(xs):
    """The second largest DISTINCT value in xs, or None if there is none."""
    distinct = sorted(set(xs))
    if len(distinct) < 2:
        return None
    return xs[-2]
