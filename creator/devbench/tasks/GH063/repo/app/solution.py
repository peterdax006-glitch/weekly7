def title_words(s):
    """Capitalise the first letter of every space-separated word and lower-case the rest."""
    return ' '.join((s[:1].upper() + w[1:].lower() for w in s.split(' ')))
