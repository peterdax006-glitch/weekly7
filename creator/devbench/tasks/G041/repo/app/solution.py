def rotate(xs, k):
    """Rotate xs right by k places (k may exceed the length; empty input stays empty)."""
    if not k:
        return []
    k = k % len(xs)
    return xs[-k:] + xs[:-k] if k else list(xs)
