"""Read single members out of the remote VCTK 0.92 zip with HTTP range requests.

Only the members asked for are downloaded (the full corpus is ~11.7 GB).
Run under scripts/lowprio.py --idle; a small sleep throttles the shared link.
"""
from __future__ import annotations

import io
import time
import zipfile

import requests

URL = "https://datashare.ed.ac.uk/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/download"
THROTTLE_S = 0.05


class RangeFile(io.RawIOBase):
    def __init__(self, url: str = URL, block: int = 1 << 15):
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        self.url = r.url  # resolved after redirect
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self.block = block
        self._cache: tuple[int, bytes] = (-1, b"")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, a: int, b: int) -> bytes:
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code in (200, 206):
                    time.sleep(THROTTLE_S)
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        raise IOError("range fetch failed")

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        start = self._cache[0]
        buf = self._cache[1]
        if not (start <= self.pos and self.pos + n <= start + len(buf)):
            want = max(n, self.block)
            end = min(self.size - 1, self.pos + want - 1)
            buf = self._fetch(self.pos, end)
            self._cache = (self.pos, buf)
            start = self.pos
        out = buf[self.pos - start: self.pos - start + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def open_zip() -> zipfile.ZipFile:
    return zipfile.ZipFile(RangeFile())
