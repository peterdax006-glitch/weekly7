def parse_int(s):
    t = s.strip()
    body = t[1:] if t[:1] in "+-" and t else t
    if not body.isdigit():
        raise ValueError(s)
    return int(t)
