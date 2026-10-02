"""Byte-level BPE, trained by Nupen itself (pure python, deterministic). Ids 0-255 are raw bytes, then merges, then <|endoftext|>."""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EOT = "<|endoftext|>"
_PRE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def pretokenize(text: str) -> list[str]:
    return _PRE.findall(text)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.ranks: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab_bytes.append(self.vocab_bytes[a] + self.vocab_bytes[b])
        self.eot_id = len(self.vocab_bytes)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.eot_id + 1

    @classmethod
    def train(cls, text: str, vocab_size: int = 4096) -> "BPETokenizer":
        text = text.replace(EOT, "\n")
        counts = Counter(pretokenize(text))
        keys = sorted(counts)
        words = [list(w.encode("utf-8")) for w in keys]
        freqs = [counts[w] for w in keys]
        pair_count: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_count[p] += freqs[i]
                where[p].add(i)
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        while len(merges) < vocab_size - 257 and heap:
            negc, pair = heapq.heappop(heap)
            cur = pair_count.get(pair, 0)
            if cur != -negc:
                if cur > 0:
                    heapq.heappush(heap, (-cur, pair))
                continue
            if cur < 2:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            touched: set[tuple[int, int]] = set()
            for i in sorted(where[pair]):
                w = words[i]
                if pair not in set(zip(w, w[1:])):
                    continue
                for p in zip(w, w[1:]):
                    pair_count[p] -= freqs[i]
                    touched.add(p)
                out: list[int] = []
                j = 0
                while j < len(w):
                    if j + 1 < len(w) and (w[j], w[j + 1]) == pair:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(w[j])
                        j += 1
                words[i] = out
                for p in zip(out, out[1:]):
                    pair_count[p] += freqs[i]
                    where[p].add(i)
                    touched.add(p)
            for p in touched:
                if pair_count[p] > 0 and p != pair:
                    heapq.heappush(heap, (-pair_count[p], p))
            pair_count[pair] = 0
        return cls(merges)

    def _encode_word(self, w: str) -> list[int]:
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        ids = list(w.encode("utf-8"))
        while len(ids) > 1:
            best = None
            best_rank = 1 << 30
            for p in zip(ids, ids[1:]):
                r = self.ranks.get(p, 1 << 30)
                if r < best_rank:
                    best_rank, best = r, p
            if best is None:
                break
            new_id = 256 + best_rank
            out: list[int] = []
            j = 0
            while j < len(ids):
                if j + 1 < len(ids) and (ids[j], ids[j + 1]) == best:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            ids = out
        self._cache[w] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for k, part in enumerate(text.split(EOT)):
            if k:
                out.append(self.eot_id)
            for w in pretokenize(part):
                out.extend(self._encode_word(w))
        return out

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for i in ids:
            if i == self.eot_id:
                buf += EOT.encode()
            elif 0 <= i < self.eot_id:
                buf += self.vocab_bytes[i]
        return buf.decode("utf-8", errors="replace")

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({"type": "byte-bpe", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([(int(a), int(b)) for a, b in d["merges"]])
