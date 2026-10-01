import math

_AREA = {
    "circle": lambda d: math.pi * d["r"] ** 2,
    "square": lambda d: d["side"] ** 2,
    "rect": lambda d: d["w"] * d["h"],
}


def area(kind, **d):
    fn = _AREA.get(kind)
    if fn is None:
        raise ValueError(kind)
    return fn(d)
