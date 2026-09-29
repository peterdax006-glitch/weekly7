"""Bible Phase 29 - dependency-free inline SVG charts for the public pages.

Each function returns an SVG/HTML string, uses CSS variables from site_render (light and dark), and degrades to a short
text note when there is nothing to draw (empty input, all-NaN, zero range) instead of emitting a broken graphic."""
import math

from scripts.site_render import esc


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _scale(lo, hi, a, b):
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    return lambda v: a + (v - lo) / (hi - lo) * (b - a)


def sparkline(values, w=120, h=26, zero=True, cls="var(--s1)"):
    """Tiny line of a series; a zero baseline is drawn when the series crosses zero."""
    v = [x for x in values if _finite(x)]
    if len(v) < 2:
        return '<span class="muted">n/a</span>'
    lo, hi = min(v), max(v)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    X, Y = _scale(0, len(v) - 1, 2, w - 2), _scale(lo, hi, h - 2, 2)
    pts = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v))
    base = f'<line x1="0" x2="{w}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis, var(--grid))" stroke-dasharray="2 2"/>' if lo < 0 < hi else ""
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="trend over {len(v)} points" '
            f'style="display:inline-block;width:{w}px">{base}<polyline fill="none" stroke="{cls}" stroke-width="1.6" points="{pts}"/></svg>')


def sign_bars(values, w=160, h=28):
    """One thin bar per period, up (good) or down (bad); shows whether an effect has the same sign in every era."""
    v = [x if _finite(x) else 0.0 for x in values]
    if not v:
        return '<span class="muted">n/a</span>'
    m = max(abs(x) for x in v) or 1.0
    bw = w / len(v)
    mid = h / 2
    bars = []
    for i, x in enumerate(v):
        bh = abs(x) / m * (mid - 1)
        y = mid - bh if x >= 0 else mid
        bars.append(f'<rect x="{i * bw + 0.3:.2f}" y="{y:.2f}" width="{max(bw - 0.8, 0.6):.2f}" height="{max(bh, 0.4):.2f}" '
                    f'fill="{"var(--good)" if x >= 0 else "var(--bad)"}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="per-year sign pattern" '
            f'style="display:inline-block;width:{w}px"><line x1="0" x2="{w}" y1="{mid}" y2="{mid}" stroke="var(--grid)"/>{"".join(bars)}</svg>')


def stacked_bar(parts, total=None):
    """parts: list of (label, value, css colour). Returns a bar with a title per segment (hover) and a text legend."""
    parts = [(l, v, c) for l, v, c in parts if _finite(v) and v > 0]
    tot = total if total else sum(v for _, v, _ in parts)
    if not tot:
        return '<span class="muted">nothing recorded</span>'
    segs = "".join(f'<i style="width:{v / tot * 100:.2f}%;background:{c}" title="{esc(l)}: {v:g}"></i>' for l, v, c in parts)
    legend = " ".join(f'<span class="note"><span style="color:{c}">■</span> {esc(l)} {v:g}</span>' for l, v, c in parts)
    return f'<div class="bar" role="img" aria-label="composition">{segs}</div><div>{legend}</div>'


def interval_plot(rows, unit="", width=640, row_h=22, fmt=lambda x: f"{x:+.4f}", label_w=170, zero_label="no effect"):
    """Forest plot. rows: dicts with label, est, lo, hi, kind in {'selected','good','bad','flat'}. The interval is drawn
    only when lo/hi are finite, so a point estimate with no uncertainty is visibly a bare dot."""
    rows = [r for r in rows if _finite(r.get("est"))]
    if not rows:
        return '<p class="muted">No tested values to plot.</p>'
    vals = [r["est"] for r in rows] + [r[k] for r in rows for k in ("lo", "hi") if _finite(r.get(k))] + [0.0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 0.5
    X = _scale(lo - pad, hi + pad, label_w, width - 12)
    h = row_h * len(rows) + 28
    col = {"selected": "var(--s2)", "good": "var(--good)", "bad": "var(--bad)", "flat": "var(--muted)"}
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="effect of each tested value with uncertainty">',
           f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="0" y2="{h - 22}" stroke="var(--ink-2)" stroke-dasharray="3 3"/>',
           f'<text x="{X(0):.1f}" y="{h - 8}" font-size="10" text-anchor="middle" fill="var(--muted)">{esc(zero_label)}</text>']
    for i, r in enumerate(rows):
        y = i * row_h + row_h / 2 + 4
        c = col.get(r.get("kind", "flat"), "var(--muted)")
        if _finite(r.get("lo")) and _finite(r.get("hi")):
            out.append(f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2" opacity=".55"/>')
        out.append(f'<circle cx="{X(r["est"]):.1f}" cy="{y}" r="{5 if r.get("kind") == "selected" else 3.6}" fill="{c}">'
                   f'<title>{esc(r.get("label"))}: {esc(fmt(r["est"]))}{esc(unit)}</title></circle>')
        out.append(f'<text x="{label_w - 8}" y="{y + 3.5}" font-size="11" text-anchor="end" fill="var(--ink-2)">{esc(str(r.get("label"))[:30])}</text>')
    out.append("</svg>")
    return "".join(out)


def calibration_dots(bins, width=320, height=180, title="claimed vs observed"):
    """bins: (claimed, observed, n). Diagonal is perfect calibration; dot area grows with n. Used for the P(real) check."""
    b = [(c, o, n) for c, o, n in bins if _finite(c) and _finite(o) and n]
    if not b:
        return '<p class="muted">No calibration bins recorded.</p>'
    X, Y = _scale(0, 1, 34, width - 10), _scale(0, 1, height - 24, 8)
    nmax = max(n for _, _, n in b)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
           f'<line x1="{X(0):.0f}" y1="{Y(0):.0f}" x2="{X(1):.0f}" y2="{Y(1):.0f}" stroke="var(--muted)" stroke-dasharray="4 3"/>',
           f'<text x="{X(0.5):.0f}" y="{height - 6}" font-size="10" text-anchor="middle" fill="var(--muted)">claimed P(real)</text>',
           f'<text x="10" y="{Y(0.5):.0f}" font-size="10" fill="var(--muted)" transform="rotate(-90 10 {Y(0.5):.0f})" text-anchor="middle">observed</text>']
    for c, o, n in b:
        r = 3 + 6 * math.sqrt(n / nmax)
        bad = o < c - 0.15
        out.append(f'<circle cx="{X(c):.1f}" cy="{Y(o):.1f}" r="{r:.1f}" fill="{"var(--bad)" if bad else "var(--s1)"}" opacity=".8">'
                   f'<title>claimed {c:.2f}, observed {o:.2f}, n={n}</title></circle>')
    out.append("</svg>")
    return "".join(out)
