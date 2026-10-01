def top_n(scores, n):
    """Names of the n highest scores from a dict name->score, highest first, ties by name."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [kv for name, _ in ranked[:n]]
