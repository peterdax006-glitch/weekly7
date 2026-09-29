"""Trust-table persistence, drift monitoring and reports (Bible PHASE 12; canon: recent relevance - a type's
reliability is not permanent, so versions are kept and compared).

TrustStore   directory of immutable versions v0001_<asof>.json, each with a content hash and its parent's hash
             (a hash chain: an edited or deleted middle version is detected on load).
diff()       cell-level changes between two tables: added, removed, reliability gained/lost, unchanged.
drift()      the monitor: flags a (type, indicator) whose posterior moved by more than z_move combined standard
             deviations, whose sign flipped, or whose reliability was gained/lost. The move bar rises with the number
             of compared cells (Bonferroni).
render_html  self-contained HTML/SVG report: heat-map of posterior reliability by SIC division x indicator, the
             reliable-share table, and the drift findings."""
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .trust import TrustTable

_NAME = re.compile(r"^v(\d{4})_(\d{4}-\d{2}-\d{2})\.json$")
_COLS = ["type", "indicator", "change", "post_a", "post_b", "sd_a", "sd_b", "reliable_a", "reliable_b"]


def _content_hash(rec: dict) -> str:
    body = {k: rec[k] for k in ("params", "now", "indicators", "table")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class StoreError(RuntimeError):
    pass


class TrustStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def versions(self) -> list:
        out = []
        for p in sorted(self.root.glob("v*.json")):
            m = _NAME.match(p.name)
            if m:
                out.append((int(m.group(1)), m.group(2), p))
        return out

    def save(self, tt: TrustTable, note="") -> int:
        """Write the next version. Refuses an empty table (nothing to version) and one identical to the latest."""
        if tt.table.empty or tt.now is None:
            raise StoreError("refusing to save an empty trust table")
        rec = dict(params=tt.p, now=str(tt.now), indicators=tt.indicators,
                   table=json.loads(tt.table.to_json(orient="records")))
        h = _content_hash(rec)
        vs = self.versions()
        prev_hash = None
        if vs:
            last = json.loads(vs[-1][2].read_text(encoding="utf-8"))
            if last["hash"] == h:
                raise StoreError(f"identical to v{vs[-1][0]:04d}; not saved")
            prev_hash = last["hash"]
        n = (vs[-1][0] + 1) if vs else 1
        rec.update(version=n, hash=h, parent=prev_hash, note=note)
        path = self.root / f"v{n:04d}_{pd.Timestamp(tt.now).date()}.json"
        path.write_text(json.dumps(rec, default=str), encoding="utf-8")
        return n

    def load(self, version=None) -> TrustTable:
        vs = self.versions()
        if not vs:
            raise StoreError("no versions stored")
        self.verify()
        pick = vs[-1] if version is None else next((v for v in vs if v[0] == version), None)
        if pick is None:
            raise StoreError(f"version {version} not found")
        rec = json.loads(pick[2].read_text(encoding="utf-8"))
        tt = TrustTable(**rec["params"])
        tt.now = pd.Timestamp(rec["now"])
        tt.indicators = rec["indicators"]
        tt.table = pd.DataFrame(rec["table"])
        tt._reindex()
        return tt

    def verify(self):
        """Recompute every hash and check the parent chain; raises StoreError on any break."""
        prev, expect = None, 1
        for n, _, p in self.versions():
            rec = json.loads(p.read_text(encoding="utf-8"))
            if n != expect:
                raise StoreError(f"version gap: expected v{expect:04d}, found v{n:04d}")
            if _content_hash(rec) != rec["hash"]:
                raise StoreError(f"v{n:04d} content does not match its hash (edited)")
            if rec["parent"] != prev:
                raise StoreError(f"v{n:04d} parent hash does not match v{n - 1:04d}")
            prev, expect = rec["hash"], n + 1
        return True


def diff(a: TrustTable, b: TrustTable) -> pd.DataFrame:
    """Cell-level change list from a to b. change in {added, removed, gained_trust, lost_trust, unchanged}."""
    A = a.table.set_index(["type", "indicator"]) if len(a.table) else pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []]))
    B = b.table.set_index(["type", "indicator"]) if len(b.table) else pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []]))
    rows = []
    for k in A.index.union(B.index):
        ia, ib = k in A.index, k in B.index
        ra = A.loc[k] if ia else None
        rb = B.loc[k] if ib else None
        if ia and not ib:
            ch = "removed"
        elif ib and not ia:
            ch = "added"
        elif ra["reliable"] and not rb["reliable"]:
            ch = "lost_trust"
        elif rb["reliable"] and not ra["reliable"]:
            ch = "gained_trust"
        else:
            ch = "unchanged"
        rows.append(dict(type=k[0], indicator=k[1], change=ch,
                         post_a=ra["post"] if ia else np.nan, post_b=rb["post"] if ib else np.nan,
                         sd_a=ra["post_sd"] if ia else np.nan, sd_b=rb["post_sd"] if ib else np.nan,
                         reliable_a=bool(ra["reliable"]) if ia else False,
                         reliable_b=bool(rb["reliable"]) if ib else False))
    out = pd.DataFrame(rows, columns=_COLS)
    out["delta"] = out["post_b"] - out["post_a"]
    return out


def drift(a: TrustTable, b: TrustTable, z_move=3.0, min_abs=0.01) -> pd.DataFrame:
    """Monitor. Cells present in both versions are flagged when
       lost_trust / gained_trust  - reliability changed, or
       sign_flip                  - both posteriors have |post| >= min_abs and opposite signs and either was reliable, or
       moved                      - |delta| exceeds max(z_move, Bonferroni bar) combined posterior sds and either was
                                    reliable.
    Successive versions share data, so `moved` is an alarm to inspect, not a p-value."""
    cols = _COLS + ["delta", "zmove", "flag"]
    both = diff(a, b).dropna(subset=["post_a", "post_b"]).copy()
    if both.empty:
        return pd.DataFrame(columns=cols)
    both["zmove"] = both["delta"] / np.sqrt(both["sd_a"] ** 2 + both["sd_b"] ** 2)
    bar = max(z_move, float(stats.norm.isf(0.025 / len(both))))
    trusted = both["reliable_a"] | both["reliable_b"]
    flip = ((np.sign(both["post_a"]) != np.sign(both["post_b"])) & (both["post_a"].abs() >= min_abs)
            & (both["post_b"].abs() >= min_abs))
    both["flag"] = np.where(both["change"].isin(["lost_trust", "gained_trust"]), both["change"],
                            np.where(flip & trusted, "sign_flip",
                                     np.where((both["zmove"].abs() >= bar) & trusted, "moved", "")))
    out = both[both["flag"] != ""].sort_values("zmove", key=np.abs, ascending=False).reset_index(drop=True)
    out.attrs["z_bar"] = bar
    return out[cols]


def _color(v, vmax):
    """Diverging blue (negative) - white - orange (positive)."""
    x = float(np.clip(v / vmax, -1, 1)) if vmax > 0 else 0.0
    if x >= 0:
        r, g, b = 255, int(255 - 110 * x), int(255 - 210 * x)
    else:
        r, g, b = int(255 + 190 * x), int(255 + 110 * x), 255
    return f"rgb({r},{g},{b})"


def heatmap_svg(tt: TrustTable, level="sic", cell=(64, 26)) -> str:
    """Types (rows) x indicators (columns) at one level; unreliable cells are hatched grey (neutralized)."""
    t = tt.table[tt.table["level"] == level]
    if t.empty:
        return "<p>no cells at this level</p>"
    types, inds = sorted(t["type"].unique()), list(dict.fromkeys(t["indicator"]))
    cw, ch = cell
    left, top = 90, 80
    vmax = float(t["post"].abs().max()) or 1.0
    lk = t.set_index(["type", "indicator"])
    w, h = left + cw * len(inds) + 10, top + ch * len(types) + 10
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
         '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<rect width="6" height="6" fill="#eee"/><line x1="0" y1="0" x2="0" y2="6" stroke="#bbb" stroke-width="2"/>'
         '</pattern></defs>']
    for j, c in enumerate(inds):
        cx = left + j * cw + cw / 2
        o.append(f'<text x="{cx}" y="{top - 8}" transform="rotate(-40 {cx} {top - 8})">{html.escape(c)}</text>')
    for i, ty in enumerate(types):
        y = top + i * ch
        o.append(f'<text x="{left - 6}" y="{y + ch / 2 + 4}" text-anchor="end">{html.escape(ty)}</text>')
        for j, c in enumerate(inds):
            x = left + j * cw
            if (ty, c) not in lk.index:
                o.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="none" stroke="#ddd"/>')
                continue
            r = lk.loc[(ty, c)]
            fill = _color(r["post"], vmax) if r["reliable"] else "url(#hatch)"
            status = "reliable" if r["reliable"] else html.escape(str(r["reason"]))
            o.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="{fill}"><title>{html.escape(ty)} / '
                     f'{html.escape(c)}: post={r["post"]:+.4f} z={r["z"]:+.1f} weeks={int(r["n_weeks"])} {status}</title></rect>')
            if r["reliable"]:
                o.append(f'<text x="{x + cw / 2 - 1}" y="{y + ch / 2 + 4}" text-anchor="middle">{r["post"]:+.2f}</text>')
    return "\n".join(o + ["</svg>"])


def render_html(tt: TrustTable, path, drift_df: pd.DataFrame = None, title="Per-type trust table") -> str:
    """Write a self-contained report and return the path. Text is HTML-escaped; no external assets."""
    asof = str(tt.now.date()) if tt.now is not None else "n/a"
    n_rel = int(tt.table["reliable"].sum()) if len(tt.table) else 0
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>",
             "<style>body{font-family:sans-serif;max-width:980px;margin:2em auto;padding:0 1em}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}</style>",
             f"<h1>{html.escape(title)}</h1><p>As of {html.escape(asof)}; {len(tt.table)} cells, {n_rel} reliable. "
             "Hatched = neutralized (no reliable evidence): the indicator is skipped for that type.</p>",
             "<h2>Reliability by SIC division</h2>", heatmap_svg(tt, "sic"),
             "<h2>Earned trust by level</h2>", tt.reliable_share().to_html(index=False, float_format=lambda v: f"{v:.3f}")]
    if drift_df is not None:
        parts.append("<h2>Drift since previous version</h2>")
        if drift_df.empty:
            parts.append("<p>No drift flagged.</p>")
        else:
            parts.append(drift_df[["type", "indicator", "flag", "post_a", "post_b", "zmove"]].round(4).to_html(index=False))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")
    return str(path)
