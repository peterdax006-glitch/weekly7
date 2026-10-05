"""Download chosen VCTK speakers (mic1, only their files) and write Piper training data.

Output under OUT:
  wav/<spk>_<utt>.wav        22.05 kHz mono 16-bit (resampled from 48 kHz FLAC)
  metadata_<spk>.csv         LJSpeech/Piper style:  file.wav|text            (single speaker)
  metadata_multi.csv         file.wav|speaker|text                           (multi speaker)
  manifest.json              counts, durations, skipped files
Resumable: existing wavs are skipped. Run at idle priority; throttled by vctk_remote.
Usage: python prepare_piper_data.py OUT --speakers p226:400 p232:150
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

MIN_S, MAX_S = 1.0, 15.0


def clean_text(t: str) -> str:
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--speakers", nargs="+", required=True, help="id[:max_utts]")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    z = vctk_remote.open_zip()
    names = set(z.namelist())
    manifest: dict = {"speakers": {}, "format": "22050 Hz mono 16-bit PCM wav; csv delimiter '|'"}
    multi = []
    for spec in a.speakers:
        spk, _, mx = spec.partition(":")
        mx = int(mx) if mx else 10 ** 9
        utts = sorted({m.group(1) for n in names if (m := re.search(rf"{spk}_(\d+)_mic1\.flac$", n))})[:mx]
        rows, dur, skipped = [], 0.0, []
        for u in utts:
            wav = out / "wav" / f"{spk}_{u}.wav"
            tkey = f"txt/{spk}/{spk}_{u}.txt"
            if tkey not in names:
                skipped.append((u, "no_text"))
                continue
            text = clean_text(z.read(tkey).decode("utf-8", "ignore"))
            if not wav.exists():
                x, sr = sf.read(io.BytesIO(z.read(f"wav48_silence_trimmed/{spk}/{spk}_{u}_mic1.flac")), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                d = len(x) / sr
                if not (MIN_S <= d <= MAX_S) or not text:
                    skipped.append((u, f"dur_{d:.1f}"))
                    continue
                y = resample_poly(x, 147, 320)  # 48000 -> 22050
                y = np.clip(y, -1, 1)
                sf.write(wav, y, 22050, subtype="PCM_16")
            info = sf.info(wav)
            dur += info.duration
            rows.append((wav.name, text))
            if len(rows) % 25 == 0:
                print(spk, len(rows), f"{dur / 60:.1f} min", flush=True)
        (out / f"metadata_{spk}.csv").write_text("".join(f"{w}|{t}\n" for w, t in rows), encoding="utf-8")
        multi += [(w, spk, t) for w, t in rows]
        manifest["speakers"][spk] = {"utts": len(rows), "minutes": round(dur / 60, 1), "skipped": skipped}
    (out / "metadata_multi.csv").write_text("".join(f"{w}|{s}|{t}\n" for w, s, t in multi), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: {"utts": v["utts"], "minutes": v["minutes"]} for k, v in manifest["speakers"].items()}))


if __name__ == "__main__":
    main(sys.argv[1:])
