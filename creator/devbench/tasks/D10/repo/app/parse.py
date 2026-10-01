def parse_int(s):
    if not s.isdigit():
        raise ValueError(s)
    return int(s)
