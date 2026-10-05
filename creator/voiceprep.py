"""Voice preparation for Phase 2 TTS (blueprint G3): a consented friend's phone clips -> a Piper/LJSpeech-style dataset.

Two halves, both CPU only and cheap:
  * receive: store_upload() keeps the exact uploaded bytes in <phone>/voice_inbox and records the speaker's consent.
  * prepare: process_inbox() decodes (ffmpeg), splits into 1-15 s utterances by voice activity, drops silence / clipping / music /
    noisy pieces, flags pieces that sound like a different speaker, transcribes locally (faster-whisper if installed, else
    whisper.cpp if a binary is on PATH, else leaves the text empty) and writes wavs/ + metadata.csv + REPORT.json.
Audio and transcripts are data: they live under the runtime directory, never in the repo.
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

SR = 22050
AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".aac", ".caf")
VIDEO_EXTS = (".mov", ".mp4", ".m4v", ".3gp")        # the audio track is extracted with ffmpeg; the uploaded bytes are stored as they came
EXTS = AUDIO_EXTS + VIDEO_EXTS
MAX_FILE = 200 * 1024 * 1024
MAX_VIDEO = 500 * 1024 * 1024
CONSENT_FLAG = "friend-agreed"
MIN_UTT, MAX_UTT = 1.0, 15.0
MIN_SNR_DB = 12.0
MAX_CLIP_FRAC = 0.005
MIN_MODULATION = 0.35        # speech energy swings between syllables; a tone or steady noise does not
F0_TOL = 0.30                # relative pitch distance from the median that flags another speaker
TIMBRE_MARGIN = 3.0          # robust z-score of the timbre distance
USABLE_MIN_FOR_FRIEND = 10.0


# ---------------------------------------------------------------- receiving

def safe_name(name: str) -> str:
    base = Path(str(name).replace("\\", "/")).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:60] or "clip"
    return stem + ext


def safe_speaker(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", str(label or "friend"))[:24] or "friend"


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], Optional[tuple[str, bytes]]]:
    """Fields and the first file part of a multipart/form-data body. File bytes are returned untouched."""
    m = re.search(r'boundary="?([^";]+)"?', content_type, re.I)
    if not m:
        raise ValueError("no multipart boundary")
    delim = b"--" + m.group(1).encode()
    fields: dict[str, str] = {}
    file: Optional[tuple[str, bytes]] = None
    for part in body.split(delim)[1:]:
        if part.startswith(b"--"):
            break
        part = part[2:] if part.startswith(b"\r\n") else part
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        h = head.decode("utf-8", "replace")
        name = re.search(r'name="([^"]*)"', h)
        fn = re.search(r'filename="([^"]*)"', h)
        if fn is not None:
            if file is None:
                file = (fn.group(1), data)
        elif name:
            fields[name.group(1)] = data.decode("utf-8", "replace").strip()
    return fields, file


def duration_seconds(p: Path) -> float:
    """Length of an audio file via ffprobe; 0.0 when it cannot be read."""
    probe = _tool("ffprobe")
    if not probe:
        return 0.0
    try:
        r = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(p)],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip() or 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def spoken_minutes(minutes: float) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    n = int(round(minutes))
    if minutes < 0.5:
        return "Less than a minute of speech so far."
    w = words[n] if n < len(words) else str(n)
    return f"{w.capitalize()} minute{'s' if n != 1 else ''} of speech so far."


def has_consent(rt: Path, speaker: str) -> bool:
    return speaker in _read_json(rt / "voice_inbox" / "CONSENT.json", {})


def record_consent(rt: Path, speaker: str, statement: str = "", now: Optional[datetime] = None) -> None:
    """Writes the speaker's consent (label, date, the owner's statement) to voice_inbox/CONSENT.json; an existing record is kept."""
    inbox = rt / "voice_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    cpath = inbox / "CONSENT.json"
    consent = _read_json(cpath, {})
    if speaker not in consent:
        consent[speaker] = {"speaker": speaker, "date": (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                            "owner_statement": (statement or "Owner states the speaker agreed to this use of their voice.")[:300], "flag": CONSENT_FLAG}
        cpath.write_text(json.dumps(consent, indent=1), encoding="utf-8")


def store_upload(rt: Path, filename: str, data: bytes, fields: dict[str, str],
                 now: Optional[datetime] = None) -> tuple[int, dict[str, Any]]:
    """Validate, record consent, store the exact bytes. Returns (http status, body). Never transcodes."""
    inbox = rt / "voice_inbox"
    now = now or datetime.now()
    ext = Path(safe_name(filename)).suffix
    if ext not in EXTS:
        return 415, {"error": "unsupported type; send " + "/".join(e[1:] for e in EXTS)}
    if not data:
        return 400, {"error": "empty file"}
    if len(data) > (MAX_VIDEO if ext in VIDEO_EXTS else MAX_FILE):
        return 413, {"error": "file too large (200 MB cap for audio, 500 MB for video)"}
    speaker = safe_speaker(fields.get("speaker", ""))
    inbox.mkdir(parents=True, exist_ok=True)
    if not has_consent(rt, speaker):
        if fields.get("consent", "").strip().lower() != CONSENT_FLAG:
            return 403, {"error": "consent required: the first upload for each speaker needs consent=" + CONSENT_FLAG}
        record_consent(rt, speaker, fields.get("statement") or "", now)
    name = f"{now.strftime('%Y%m%d-%H%M%S-%f')}_{speaker}_{safe_name(filename)}"
    dest = inbox / name
    dest.write_bytes(data)
    durs = _read_json(inbox / "DURATIONS.json", {})
    durs[name] = round(duration_seconds(dest), 2)
    (inbox / "DURATIONS.json").write_text(json.dumps(durs, indent=1), encoding="utf-8")
    minutes = sum(durs.values()) / 60.0
    return 200, {"ok": True, "stored": name, "bytes": len(data), "minutes_received": round(minutes, 2),
                 "reply": "Received, sir. " + spoken_minutes(minutes)}


_NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
_MEDIA_VERB = r"(?:send|upload|pull|grab|get|fetch|take|give|share|use|accept|receive|got)"
MEDIA_REQ = re.compile(r"(?:" + _MEDIA_VERB + r".{0,70}" + r"(?:voice|videos?|photos?|pictures?|album|recordings?|memos?|clips?|samples?)"
                       r"|(?:videos?|photos?|album|voice).{0,60}" + _MEDIA_VERB + r")", re.I)
_MEDIA_NOUN = re.compile(r"(?:voice|videos?|photos?|pictures?|album|recordings?|memos?|clips?|samples?)", re.I)
DEFAULT_COUNT = 5
LATEST_CAP = 50


def parse_media_request(text: str) -> Optional[dict[str, Any]]:
    """Owner words -> the send_media action parameters, or None when it is not a request to pull media from the phone.
    'my last 3 videos' -> latest, video, 3; 'the latest video' -> 1; 'everything in my Nupen Voice album' -> album, count 0 (all);
    'the videos from today' -> latest, since today. Plural without a number: the last 5."""
    t = text.strip()
    if not MEDIA_REQ.search(t) or not _MEDIA_NOUN.search(t):
        return None
    low = t.lower()
    kind = "video" if re.search(r"videos?", low) else ("photo" if re.search(r"photos?|pictures?", low) else "any")
    m = re.search(r"(?:in|from|out of)\s+(?:my\s+|the\s+)?(.+?)\s+album|album\s+(?:called|named)\s+[\"']?([^\"']+?)[\"']?\s*$", t, re.I)
    album = (m.group(1) or m.group(2)).strip(" .,'\"") if m else ""
    since = "today" if re.search(r"\b(?:from )?today(?:'s)?\b|\bthis morning\b|\bthis afternoon\b", low) else ""
    n = None
    cm = re.search(r"\b(?:last|latest|recent|newest|past|previous|first|top)\s+(\d+|" + "|".join(_NUMS) + r")\b", low) or re.search(r"\b(\d+)\s+(?:videos?|photos?|pictures?|clips?|recordings?)", low)
    if cm:
        g = cm.group(1)
        n = int(g) if g.isdigit() else _NUMS[g]
    elif re.search(r"\b(?:the |my )?(?:last|latest|newest|most recent)\s+(?:video|photo|picture|clip|recording|memo)\b|\b(?:this|that|the) (?:video|photo|clip)\b", low):
        n = 1
    if album:
        count = 0 if re.search(r"\b(?:everything|all|every)\b", low) or n is None else n
        source = "album"
    else:
        count = min(n, LATEST_CAP) if n else (LATEST_CAP if since else DEFAULT_COUNT)
        source = "latest"
    return {"source": source, "kind": kind, "count": count, "album": album if source == "album" else "", "since": since}


# ---------------------------------------------------------------- audio helpers

def _tool(name: str) -> Optional[str]:
    p = shutil.which(name)
    if p:
        return p
    base = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for c in base.glob(f"Gyan.FFmpeg*/*/bin/{name}.exe") if base.exists() else []:
        return str(c)
    return None


def decode(src: Path, dst: Path) -> str:
    """Any supported file -> mono 22.05 kHz 16-bit wav. Returns the tool used."""
    ff = _tool("ffmpeg")
    if ff:
        r = subprocess.run([ff, "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dst)],
                           capture_output=True, timeout=1800)
        if r.returncode != 0 or not dst.exists():
            raise RuntimeError("ffmpeg failed: " + r.stderr.decode("utf-8", "replace")[:200])
        return "ffmpeg"
    if src.suffix.lower() == ".wav":
        x, sr = read_wav(src)
        write_wav(dst, _resample(x, sr))
        return "python-wav"
    raise RuntimeError("no ffmpeg and no Python decoder for " + src.suffix)


def read_wav(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p), "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError("unsupported wav width")
    return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), sr


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def write_wav(p: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- segmentation and quality

FRAME = int(0.02 * SR)


def _frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0)
    f = x[: n * FRAME].reshape(n, FRAME)
    return 10 * np.log10(np.mean(f * f, axis=1) + 1e-10)


def split_utterances(x: np.ndarray) -> list[tuple[int, int]]:
    """Voice-activity split on frame energy with an adaptive threshold; pieces of MIN_UTT..MAX_UTT seconds (sample ranges)."""
    db = _frame_db(x)
    if len(db) < 10:
        return []
    floor, top = np.percentile(db, 10), np.percentile(db, 95)
    if top - floor < 10:                      # no real contrast between speech and pause: nothing to split on
        return [(0, len(x))] if len(x) / SR <= MAX_UTT else _chunk(0, len(x), db)
    thr = floor + 0.35 * (top - floor)
    act = db > thr
    gap = 12                                  # 240 ms of quiet ends a phrase
    segs, start, quiet = [], None, 0
    for i, a in enumerate(act):
        if a:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap:
                segs.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        segs.append((start, len(act) - quiet))
    out: list[tuple[int, int]] = []
    pad = 5
    cur: Optional[list[int]] = None
    for s, e in segs:                         # glue short phrases together up to MAX_UTT
        if cur is None:
            cur = [s, e]
        elif (e - cur[0]) * FRAME / SR <= MAX_UTT and (s - cur[1]) * FRAME / SR < 0.8 and (cur[1] - cur[0]) * FRAME / SR < MIN_UTT * 3:
            cur[1] = e
        else:
            out.append((cur[0], cur[1]))
            cur = [s, e]
    if cur:
        out.append((cur[0], cur[1]))
    res: list[tuple[int, int]] = []
    for s, e in out:
        a, b = max(0, s - pad) * FRAME, min(len(db), e + pad) * FRAME
        if (b - a) / SR > MAX_UTT:
            res.extend(_chunk(a, b, db))
        else:
            res.append((a, b))
    return [(a, b) for a, b in res if (b - a) / SR >= MIN_UTT]


def _chunk(a: int, b: int, db: np.ndarray) -> list[tuple[int, int]]:
    """Cut a long span at its quietest frames so no piece exceeds MAX_UTT."""
    out, cur = [], a
    maxf = int(MAX_UTT * SR / FRAME) - 2
    while (b - cur) // FRAME > maxf:
        lo, hi = cur // FRAME + int(maxf * 0.5), cur // FRAME + maxf
        cut = lo + int(np.argmin(db[lo:hi]))
        out.append((cur, cut * FRAME))
        cur = cut * FRAME
    out.append((cur, b))
    return out


def quality(x: np.ndarray) -> dict[str, float]:
    db = _frame_db(x)
    if len(db) < 5:
        return {"snr_db": 0.0, "clip": 0.0, "modulation": 0.0}
    noise = np.percentile(db, 10)
    speech = np.percentile(db, 90)
    lin = 10 ** (db / 10)
    mod = float(np.std(lin) / (np.mean(lin) + 1e-12))
    return {"snr_db": float(speech - noise), "clip": float(np.mean(np.abs(x) > 0.985)), "modulation": mod}


def _f0(x: np.ndarray) -> float:
    """Median pitch (Hz) of the voiced frames by autocorrelation; 0 if none."""
    n = int(0.04 * SR)
    lo, hi = int(SR / 400), int(SR / 70)
    vals = []
    for i in range(0, len(x) - n, n):
        f = x[i:i + n] - np.mean(x[i:i + n])
        if np.mean(f * f) < 1e-6:
            continue
        ac = np.correlate(f, f, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / ac[0] > 0.45:
            vals.append(SR / k)
    return float(np.median(vals)) if len(vals) >= 3 else 0.0


def _timbre(x: np.ndarray) -> np.ndarray:
    """Mean log spectrum in 24 bands (a cheap voice-colour fingerprint), mean removed."""
    n = 512
    w = np.hanning(n)
    frames = [x[i:i + n] * w for i in range(0, len(x) - n, n // 2)]
    if not frames:
        return np.zeros(24)
    p = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    edges = np.unique(np.geomspace(2, len(p) - 1, 25).astype(int))
    b = np.array([np.mean(p[edges[i]:max(edges[i] + 1, edges[i + 1])]) for i in range(len(edges) - 1)])
    lg = np.log10(b + 1e-12)
    out = np.zeros(24)
    out[: len(lg)] = lg
    return out - out.mean()


def flag_other_speakers(pieces: list[dict[str, Any]], feats: list[tuple[float, np.ndarray]]) -> list[bool]:
    if len(pieces) < 4:
        return [False] * len(pieces)
    f0s = np.array([f[0] for f in feats])
    voiced = f0s[f0s > 0]
    med_f0 = float(np.median(voiced)) if len(voiced) else 0.0
    tim = np.stack([f[1] for f in feats])
    centre = np.median(tim, axis=0)
    dist = np.linalg.norm(tim - centre, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) + 1e-6
    flags = []
    for i, (f0, _) in enumerate(feats):
        off_pitch = med_f0 > 0 and f0 > 0 and abs(f0 - med_f0) / med_f0 > F0_TOL
        off_timbre = (dist[i] - med) / (1.4826 * mad) > TIMBRE_MARGIN and dist[i] > 1.5 * med
        flags.append(bool(off_pitch and off_timbre) or bool(off_timbre and dist[i] > 2.5 * med))
    return flags


# ---------------------------------------------------------------- transcription

Transcriber = Callable[[Path], str]


def default_transcriber(model: str = "base.en") -> tuple[Optional[Transcriber], str]:
    """faster-whisper on CPU (int8) if importable, else a whisper.cpp CLI on PATH, else None."""
    try:
        from faster_whisper import WhisperModel          # type: ignore
        m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=2)

        def fw(p: Path) -> str:
            segs, _ = m.transcribe(str(p), language="en", beam_size=1, vad_filter=False)
            return " ".join(s.text.strip() for s in segs).strip()
        return fw, "faster-whisper:" + model
    except Exception:                                    # noqa: BLE001 - missing package or no model download: fall through
        pass
    cli = shutil.which("whisper-cli") or shutil.which("main")
    mp = Path.home() / "whisper.cpp" / f"ggml-{model}.bin"
    if cli and mp.exists():
        def wc(p: Path) -> str:
            r = subprocess.run([cli, "-m", str(mp), "-f", str(p), "-nt", "-np"], capture_output=True, text=True, timeout=600)
            return r.stdout.strip()
        return wc, "whisper.cpp:" + model
    return None, "none"


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("|", " ")).strip()


# ---------------------------------------------------------------- the pipeline

def process_inbox(rt: Path, out_dir: Path, transcribe: Optional[Transcriber] = None, tool_label: str = "",
                  speaker: str = "friend") -> dict[str, Any]:
    """Process every inbox file not yet in the manifest; rebuild metadata.csv and REPORT.json. Idempotent."""
    inbox = rt / "voice_inbox"
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    man = _read_json(mpath, {"files": {}, "pieces": []})
    if transcribe is None and not tool_label:
        transcribe, tool_label = default_transcriber()
    tool_label = tool_label or "custom"
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    decoder = ""
    todo = sorted(p for p in inbox.glob("*") if p.suffix.lower() in EXTS and p.name not in man["files"]
                  and f"_{speaker}_" in p.name)
    for src in todo:
        entry: dict[str, Any] = {"decoder": "", "error": ""}
        try:
            full = tmp / (src.stem + ".wav")
            entry["decoder"] = decoder = decode(src, full)
            x, sr = read_wav(full)
            full.unlink()
            spans = split_utterances(x)
            entry["spans"] = len(spans)
            pieces, feats = [], []
            for k, (a, b) in enumerate(spans):
                seg = x[a:b]
                q = quality(seg)
                pid = f"{src.stem[:15].replace('-', '')}_{k:03d}"
                piece: dict[str, Any] = {"id": pid, "file": src.name, "seconds": round(len(seg) / SR, 2), "reason": "", "text": "", **{k2: round(v, 3) for k2, v in q.items()}}
                if q["clip"] > MAX_CLIP_FRAC:
                    piece["reason"] = "clipping"
                elif q["snr_db"] < MIN_SNR_DB:
                    piece["reason"] = "noisy_or_silent"
                elif q["modulation"] < MIN_MODULATION:
                    piece["reason"] = "not_speech_or_music"
                pieces.append(piece)
                feats.append((_f0(seg), _timbre(seg)))
                piece["_seg"] = seg
            good = [i for i, p in enumerate(pieces) if not p["reason"]]
            fl = flag_other_speakers([pieces[i] for i in good], [feats[i] for i in good])
            for i, f in zip(good, fl):
                if f:
                    pieces[i]["reason"] = "different_speaker"
            for p in pieces:
                seg = p.pop("_seg")
                if not p["reason"]:
                    wp = wavs / (p["id"] + ".wav")
                    write_wav(wp, seg)
                    if transcribe is not None:
                        try:
                            p["text"] = clean_text(transcribe(wp))
                        except Exception as e:           # noqa: BLE001
                            p["text"] = ""
                            p["note"] = "transcribe failed: " + str(e)[:80]
                    if transcribe is not None and len(p["text"]) < 3:
                        p["reason"] = "no_transcript"
                        wp.unlink(missing_ok=True)
                man["pieces"].append(p)
        except Exception as e:                           # noqa: BLE001 - one bad file never stops the rest
            entry["error"] = str(e)[:200]
        man["files"][src.name] = entry
    shutil.rmtree(tmp, ignore_errors=True)
    mpath.write_text(json.dumps(man, indent=1), encoding="utf-8")
    kept = [p for p in man["pieces"] if not p["reason"]]
    with open(out_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        for p in kept:
            if p["text"]:
                f.write(f"{p['id']}|{p['text']}|{p['text']}\n")
    dropped: dict[str, int] = {}
    for p in man["pieces"]:
        if p["reason"]:
            dropped[p["reason"]] = dropped.get(p["reason"], 0) + 1
    usable = sum(p["seconds"] for p in kept) / 60.0
    report = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "speaker": speaker, "usable_minutes": round(usable, 2),
              "pieces_kept": len(kept), "pieces_dropped": sum(dropped.values()), "dropped_by_reason": dropped,
              "files_processed": len(man["files"]), "file_errors": {k: v["error"] for k, v in man["files"].items() if v.get("error")},
              "decoder": decoder or next((v["decoder"] for v in man["files"].values() if v.get("decoder")), ""),
              "transcriber": tool_label, "transcribed": sum(1 for p in kept if p["text"]),
              "enough_for_friend_voice": usable >= USABLE_MIN_FOR_FRIEND, "target_minutes": "30-60"}
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phone-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker", default="friend")
    ap.add_argument("--model", default="base.en")
    a = ap.parse_args(argv)
    t, label = default_transcriber(a.model)
    r = process_inbox(Path(a.phone_dir), Path(a.out), t, label, a.speaker)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
