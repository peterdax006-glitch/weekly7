def caesar(s, k):
    """Shift letters of s by k places in the alphabet, keeping case; other characters unchanged."""
    out = []
    for ch in s:
        if ch.isupper():
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
        elif ch.islower():
            out.append(chr((ord(ch) - 97 + k) % 26 + 97))
        else:
            out.append(ch)
    return ''.join(ch)
