def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    i = j = 0
    out = []
    while a < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
