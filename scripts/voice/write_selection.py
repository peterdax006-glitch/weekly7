"""Turn ranking.json (select_speakers.py) into SELECTION.md with numbers and the choice.

Usage: python write_selection.py WORKDIR spk1 [spk2 ...]   (chosen speakers, in order of preference)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv):
    work = Path(argv[0])
    chosen = argv[1:]
    rows = json.loads((work / "ranking.json").read_text())
    info = {}
    for line in (work / "speaker-info.txt").read_text(errors="ignore").splitlines()[1:]:
        p = line.split()
        if len(p) >= 4:
            info[p[0]] = f"{p[1]} y, {p[3]}, {' '.join(p[4:])}"
    n = len(rows)
    f0s = sorted(r["f0_median_hz"] for r in rows)
    rates = sorted(r["rate_syl_per_s"] for r in rows)
    lines = [
        "# Speaker selection for Nupen's own voice (VCTK 0.92, CC BY 4.0)", "",
        f"Candidates: {n} male speakers with an English (England) accent. Scottish, Irish and other accents were excluded to keep",
        "a refined southern/RP-leaning British register. Per speaker: 8 utterances (mic1) were measured at idle priority.", "",
        "Features: median F0 by 40 ms autocorrelation on voiced frames; F0 IQR across the 8 per-utterance medians (pitch steadiness; smaller = calmer);",
        "speaking rate = transcript syllables / trimmed audio seconds (lower = more measured);",
        "spectral centroid and low/high band tilt (80 Hz-1 kHz vs 1-4 kHz energy in dB; lower centroid, higher tilt = warmer);",
        "SNR proxy = 90th/8th percentile frame energy.", "",
        "Score = 1.5 z(-F0) + 1.0 z(-rate) + 0.5 z(-F0 IQR) + 0.7 z(-centroid) + 0.4 z(tilt) + 0.3 z(SNR). Higher is calmer, lower, more measured.", "",
        f"Pool medians: F0 {f0s[n // 2]} Hz, rate {rates[n // 2]} syl/s.", "",
        "| rank | speaker | profile | F0 Hz | F0 IQR | rate syl/s | centroid Hz | tilt dB | SNR dB | score |", "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        mark = " **chosen**" if r["speaker"] in chosen else ""
        lines.append(f"| {i} | {r['speaker']}{mark} | {info.get(r['speaker'], '')} | {r['f0_median_hz']} | {r['f0_iqr_hz']} | {r['rate_syl_per_s']} | "
                     f"{r['centroid_hz']:.0f} | {r['tilt_db']} | {r['snr_db']} | {r['score']} |")
    (work / "SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main(sys.argv[1:])
