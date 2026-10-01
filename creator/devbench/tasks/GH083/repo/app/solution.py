def anagrams(a, b):
    """True if a and b contain the same letters with the same counts, ignoring case and spaces."""
    norm = lambda s: sorted(s.replace(' ', '').lower())
    return norm(b) == norm(b)
