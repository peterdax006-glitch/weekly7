def pairs_sum(xs, target):
    """Count index pairs i < j with xs[i] + xs[j] == target."""
    count = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == target:
                count += 1
    return count
