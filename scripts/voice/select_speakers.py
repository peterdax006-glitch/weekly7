"""Measure British (England) male VCTK speakers and rank them for a calm, low, measured voice.

Downloads only N short utterances per candidate speaker (range requests, idle priority).
Features per speaker: median F0 (autocorrelation), speaking rate (syllables/s from the
transcript over trimmed duration), spectral centroid and low/high band tilt (timbre warmth),
and a crude SNR. Output: features.json (+ ranking printed).
Usage: python select_speakers.py WORKDIR [--per-speaker 14]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import vctk_remote  # noqa: E402

FLAC = "wav48_silence_trimmed/{s}/{s}_{u}_mic1.flac"
TXT = "txt/{s}/{s}_{u}.txt"


def candidates(info_path: Path, accents=("English",)) -> list[str]:
    out = []
    for line in info_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        p = line.split()
        if len(p) >= 4 and p[2] == "M" and p[3] in accents:
            out.append(p[0])
    return out


def syllables(text: str) -> int:
    n = 0
    for w in re.findall(r"[a-z]+", text.lower()):
        g = re.findall(r"[aeiouy]+", w)
        c = max(1, len(g))
        if w.endswith("e") and c > 1 and not w.endswith(("le", "ee")):
            c -= 1
        n += c
    return n


def f0_median(x: np.ndarray, sr: int) -> float:
    fl, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 300), int(sr / 60)
    vals = []
    thr = 0.25 * np.max(np.abs(x))
    for i in range(0, len(x) - fl, hop):
        f = x[i:i + fl]
        if np.sqrt(np.mean(f ** 2)) < 0.05 * np.max(np.abs(x)) or np.max(np.abs(f)) < thr * 0.4:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[fl - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] > 0.5:
            vals.append(sr / (lo + k))
    return float(np.median(vals)) if len(vals) > 20 else float("nan")


def timbre(x: np.ndarray, sr: int) -> tuple[float, float]:
    fl, hop = 1024, 512
    win = np.hanning(fl)
    cents, tilts = [], []
    e = np.array([np.sqrt(np.mean(x[i:i + fl] ** 2)) for i in range(0, len(x) - fl, hop)])
    if len(e) == 0:
        return float("nan"), float("nan")
    gate = 0.3 * np.median(e[e > 0]) if np.any(e > 0) else 0
    freqs = np.fft.rfftfreq(fl, 1 / sr)
    for j, i in enumerate(range(0, len(x) - fl, hop)):
        if e[j] < max(gate, 1e-4):
            continue
        s = np.abs(np.fft.rfft(x[i:i + fl] * win)) ** 2
        cents.append(float((freqs * s).sum() / s.sum()))
        lo = s[(freqs >= 80) & (freqs < 1000)].sum()
        hi = s[(freqs >= 1000) & (freqs < 4000)].sum()
        tilts.append(10 * np.log10((lo + 1e-12) / (hi + 1e-12)))
    return float(np.median(cents)), float(np.median(tilts))


def snr_db(x: np.ndarray, sr: int) -> float:
    fl = int(0.02 * sr)
    e = np.array([np.mean(x[i:i + fl] ** 2) for i in range(0, len(x) - fl, fl)]) + 1e-12
    return float(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 8)))


def measure(z, names: set[str], spk: str, n: int) -> dict:
    utts = sorted({m.group(1) for nme in names if (m := re.search(rf"{spk}_(\d+)_mic1\.flac$", nme))})
    utts = [u for u in utts if int(u) >= 4][:n]
    f0s, cents, tilts, snrs = [], [], [], []
    syl = dur = 0.0
    for u in utts:
        try:
            raw = z.read(FLAC.format(s=spk, u=u))
            txt = z.read(TXT.format(s=spk, u=u)).decode("utf-8", "ignore")
        except KeyError:
            continue
        x, sr = sf.read(io.BytesIO(raw), dtype="float32")
        x16 = resample_poly(x, 1, 3)  # 48k -> 16k
        f = f0_median(x16, 16000)
        if f == f:
            f0s.append(f)
        c, t = timbre(x16, 16000)
        cents.append(c)
        tilts.append(t)
        snrs.append(snr_db(x16, 16000))
        syl += syllables(txt)
        dur += len(x) / sr
    return {
        "speaker": spk, "utts": len(utts), "f0_median_hz": round(float(np.median(f0s)), 1) if f0s else None,
        "f0_iqr_hz": round(float(np.subtract(*np.percentile(f0s, [75, 25]))), 1) if f0s else None,
        "rate_syl_per_s": round(syl / dur, 2) if dur else None,
        "centroid_hz": round(float(np.nanmedian(cents)), 0), "tilt_db": round(float(np.nanmedian(tilts)), 2),
        "snr_db": round(float(np.median(snrs)), 1),
    }


def rank(rows: list[dict]) -> list[dict]:
    def z(key, sign):
        v = np.array([r[key] for r in rows], dtype=float)
        return sign * (v - v.mean()) / (v.std() + 1e-9)
    # calm = low pitch, measured = lower rate, narrow pitch spread; warm = low centroid, high LF tilt; clean recording
    score = 1.5 * z("f0_median_hz", -1) + 1.0 * z("rate_syl_per_s", -1) + 0.5 * z("f0_iqr_hz", -1) \
        + 0.7 * z("centroid_hz", -1) + 0.4 * z("tilt_db", +1) + 0.3 * z("snr_db", +1)
    for r, s in zip(rows, score):
        r["score"] = round(float(s), 2)
    return sorted(rows, key=lambda r: -r["score"])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--per-speaker", type=int, default=14)
    a = ap.parse_args(argv)
    work = Path(a.work)
    out = work / "features.json"
    done = {r["speaker"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    z = vctk_remote.open_zip()
    names = set(z.namelist())
    for spk in candidates(work / "speaker-info.txt"):
        if spk in done:
            continue
        done[spk] = measure(z, names, spk, a.per_speaker)
        out.write_text(json.dumps(list(done.values()), indent=1))
        print(done[spk], flush=True)
    ranked = rank([r for r in done.values() if r["f0_median_hz"]])
    (work / "ranking.json").write_text(json.dumps(ranked, indent=1))
    for r in ranked:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1:])
