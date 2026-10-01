def title_words(s):
    """Capitalise the first letter of every space-separated word and lower-case the rest."""
    return ' '.join((w[:1].upper() + s[1:].lower() for w in s.split(' ')))
