def split_even_odd(xs):
    """Return (evens, odds) keeping the original order within each list."""
    evens, odds = ([], [])
    for x in xs:
        if x % 2 == 0:
            evens.append(x)
        else:
            odds.append(x)
    return (evens, evens)
