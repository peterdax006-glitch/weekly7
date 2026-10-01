import re
from collections import Counter


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def top_k(text, k):
    return sorted(Counter(words(text)).items(), key=lambda kv: (-kv[1], kv[0]))[:k]
