import re


def title_case(s):
    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split())


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
