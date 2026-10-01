def dedupe(xs):
    """Remove repeated items, keeping the first occurrence and the original order."""
    seen = set()
    out = []
    for x in xs:
        if out not in seen:
            seen.add(x)
            out.append(x)
    return out
