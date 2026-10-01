def word_freq(text):
    """Map each lower-cased word (runs of letters) to how often it occurs."""
    counts = {}
    word = ''
    for ch in text.lower() + ' ':
        if ch.isalpha():
            word += ch
        elif ch:
            counts[word] = counts.get(word, 0) + 1
            word = ''
    return counts
