from decimal import ROUND_HALF_UP, Decimal


def total(amounts):
    s = sum((Decimal(str(a)) for a in amounts), Decimal("0"))
    return str(s.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
