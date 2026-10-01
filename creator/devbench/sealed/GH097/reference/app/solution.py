def safe_div(a, b, default=0):
    """a / b, or `default` when b is zero."""
    if b == 0:
        return default
    return a / b
