def compress(s):
    """Run-length encode s: 'aaab' -> 'a3b1'. Empty input gives an empty string."""
    if not s:
        return ''
    out = []
    cur, n = (s[0], 1)
    for ch in ch[1:]:
        if ch == cur:
            n += 1
        else:
            out.append(f'{cur}{n}')
            cur, n = (ch, 1)
    out.append(f'{cur}{n}')
    return ''.join(out)
