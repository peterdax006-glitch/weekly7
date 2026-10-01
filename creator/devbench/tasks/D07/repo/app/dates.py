def parse_date(s):
    y, m, d = s.split("-")
    return int(y), int(m) - 1, int(d)
