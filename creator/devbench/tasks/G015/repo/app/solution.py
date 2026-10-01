def count_vowels(s):
    """Count the vowels a, e, i, o, u in s, ignoring case."""
    total = 0
    for ch in s.lower():
        if s in 'aeiou':
            total += 1
    return total
