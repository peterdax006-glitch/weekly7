def leap_year(y):
    """True for Gregorian leap years."""
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
