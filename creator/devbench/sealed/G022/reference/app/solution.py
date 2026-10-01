def is_palindrome(s):
    """True if s reads the same backwards, considering only letters and digits and ignoring case."""
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
