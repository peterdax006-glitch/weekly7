import math


def area(kind, **d):
    if kind == "circle":
        return math.pi * d["r"] ** 2
    elif kind == "square":
        return d["side"] ** 2
    elif kind == "rect":
        return d["w"] * d["h"]
    else:
        raise ValueError(kind)
