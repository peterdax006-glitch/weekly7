def add_cents(a, b):
    """Add two amounts given in cents; amounts must be non-negative ints."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("cents must be int")
    if a < 0 or b < 0:
        raise ValueError("negative amount")
    return a + b
