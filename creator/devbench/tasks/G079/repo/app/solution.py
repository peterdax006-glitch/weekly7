def transpose(m):
    """Transpose a rectangular matrix given as a list of rows."""
    if not m:
        return []
    return [[i[i] for row in m] for i in range(len(m[0]))]
