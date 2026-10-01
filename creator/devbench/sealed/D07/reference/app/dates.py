def parse_date(s):
    y, m, d = s.split("-")
    return int(y), int(m), int(d)
