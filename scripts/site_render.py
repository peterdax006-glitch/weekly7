"""Bible Phase 29 - shared HTML shell and formatting for the generated public pages (mobile first, no external JS).

Every value that comes from a state file passes through `esc`; numbers go through the `fmt_*` helpers so a missing or
non-finite value renders as an em dash and never as 'nan' or 'None'. The only script on a page is a small inline
sortable/filterable table helper, so pages work from GitHub Pages, from file:// and with JavaScript blocked
(tables are server-rendered; JS only adds sorting and filtering)."""
import html
import math
from datetime import datetime, timezone

NAV = [("./", "Dashboard"), ("explorer.html", "Pattern Explorer"), ("sensitivity2.html", "Sensitivity"),
       ("runs.html", "Runs and registry"), ("checklist.html", "Checklist and unproven")]

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;
--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--good:#006300;--bad:#d03b3b;--warn:#a66300;--wash:#e9f2fd;--goodw:#e6f3e6;--badw:#fbeaea;--warnw:#fbf1dc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--bad:#e66767;--warn:#e0a030;--wash:#16263a;--goodw:#12261a;--badw:#2e1717;--warnw:#2b2312}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--bad:#e66767;--warn:#e0a030;--wash:#16263a;--goodw:#12261a;--badw:#2e1717;--warnw:#2b2312}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:20px 16px 64px}a{color:var(--s1)}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin-bottom:14px}nav a.on{font-weight:650;color:var(--ink);text-decoration:none}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}h2{font-size:17px;margin:0 0 4px}h3{font-size:14px;margin:14px 0 4px}
.sub{color:var(--ink-2);font-size:13px}.note{font-size:12px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px;margin:14px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:10px 12px}
.tile .k{color:var(--ink-2);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.tile .v{font-size:21px;font-weight:650}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--muted);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
th[data-sort]{cursor:pointer}td{padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.badge{display:inline-block;border-radius:6px;padding:1px 7px;font-size:12px;background:var(--wash);white-space:nowrap}
.b-good{background:var(--goodw);color:var(--good)}.b-bad{background:var(--badw);color:var(--bad)}.b-warn{background:var(--warnw);color:var(--warn)}
.good{color:var(--good)}.bad{color:var(--bad)}.muted{color:var(--muted)}.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0;font-size:13px}
select,input[type=search]{font:inherit;font-size:13px;padding:5px 8px;border-radius:8px;border:1px solid var(--ring);background:var(--surface);color:var(--ink);max-width:100%}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:760px){.two{grid-template-columns:1fr}}
svg{display:block;width:100%;height:auto;overflow:visible}details{margin:2px 0}summary{cursor:pointer;color:var(--s1)}
pre{white-space:pre-wrap;word-break:break-word;font-size:11px;background:var(--page);padding:8px;border-radius:8px;max-height:260px;overflow:auto}
.bar{height:10px;border-radius:5px;background:var(--grid);overflow:hidden;display:flex;min-width:80px}.bar i{display:block;height:100%}
.hist{border-left:2px solid var(--grid);margin:4px 0 4px 6px;padding-left:10px;font-size:12px}
.hist div{margin:2px 0}.hidden{display:none}
"""

JS = """
(function(){
function num(s){var x=parseFloat(String(s).replace(/[^0-9eE.+-]/g,''));return isNaN(x)?null:x}
document.querySelectorAll('table.sortable').forEach(function(t){
 t.querySelectorAll('th[data-sort]').forEach(function(th){
  th.addEventListener('click',function(){
   var idx=Array.prototype.indexOf.call(th.parentNode.children,th),dir=th.dataset.dir==='a'?-1:1;th.dataset.dir=dir===1?'a':'d';
   var body=t.tBodies[0],rows=Array.prototype.slice.call(body.rows);
   rows.sort(function(a,b){var x=a.cells[idx].dataset.v||a.cells[idx].textContent,y=b.cells[idx].dataset.v||b.cells[idx].textContent,nx=num(x),ny=num(y);
    if(nx!==null&&ny!==null)return (nx-ny)*dir;return String(x).localeCompare(String(y))*dir});
   rows.forEach(function(r){body.appendChild(r)});});});});
function apply(box){
 var t=document.getElementById(box.dataset.table),q=(box.querySelector('input[type=search]')||{}).value||'';q=q.toLowerCase();
 var sels=box.querySelectorAll('select[data-col]');
 Array.prototype.forEach.call(t.tBodies[0].rows,function(r){
  var ok=(r.textContent||'').toLowerCase().indexOf(q)>=0;
  sels.forEach(function(s){if(s.value&&ok){ok=(r.getAttribute('data-'+s.dataset.col)||'')===s.value}});
  r.classList.toggle('hidden',!ok);});
 var n=t.tBodies[0].querySelectorAll('tr:not(.hidden)').length,c=box.querySelector('.count');if(c)c.textContent=n+' shown';}
document.querySelectorAll('.controls[data-table]').forEach(function(box){
 box.addEventListener('input',function(){apply(box)});apply(box);});
})();
"""


def esc(x):
    return html.escape("" if x is None else str(x), quote=True)


def _bad(x):
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def fmt_num(x, d=3):
    return "—" if _bad(x) else f"{x:,.{d}f}"


def fmt_pct(x, d=1, sign=False):
    if _bad(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def fmt_int(x):
    return "—" if _bad(x) else f"{int(round(x)):,}"


def fmt_t(x):
    return "—" if _bad(x) else f"{x:+.2f}"


def sign_cls(x):
    return "" if _bad(x) or x == 0 else ("good" if x > 0 else "bad")


def cell(text, sort=None, cls="", **attrs):
    """A <td> whose data-v drives numeric sorting independent of formatting. `text` must already be escaped HTML."""
    a = "".join(f' data-{k}="{esc(v)}"' for k, v in attrs.items())
    dv = f' data-v="{esc(sort)}"' if sort is not None and not _bad(sort) else ""
    return f'<td class="{cls}"{dv}{a}>{text}</td>'


def badge(text, kind=""):
    return f'<span class="badge {kind}">{esc(text)}</span>'


def table(headers, rows, tid, sortable=True, row_attrs=None):
    """headers: list of (label, sortable). rows: list of lists of pre-rendered <td> strings. row_attrs: per-row dict."""
    th = "".join(f'<th{" data-sort" if s and sortable else ""}>{esc(h)}</th>' for h, s in headers)
    body = []
    for i, r in enumerate(rows):
        ra = "".join(f' data-{k}="{esc(v)}"' for k, v in (row_attrs[i] if row_attrs else {}).items())
        body.append(f"<tr{ra}>{''.join(r)}</tr>")
    if not body:
        body = [f'<tr><td colspan="{len(headers)}" class="muted">Nothing recorded yet.</td></tr>']
    return (f'<div class="scroll"><table id="{tid}" class="{"sortable" if sortable else ""}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def controls(table_id, selects=(), placeholder="Search"):
    """selects: list of (data-col name, label, [values]). Rows carry the matching data-<col> attribute."""
    sel = "".join(f'<label>{esc(lab)} <select data-col="{esc(col)}"><option value="">all</option>'
                  + "".join(f'<option value="{esc(v)}">{esc(v)}</option>' for v in vals) + "</select></label>"
                  for col, lab, vals in selects)
    return (f'<div class="controls" data-table="{esc(table_id)}"><input type="search" placeholder="{esc(placeholder)}" '
            f'aria-label="{esc(placeholder)}">{sel}<span class="note count"></span></div>')


def tile(k, v, sub=""):
    s = f'<div class="note">{esc(sub)}</div>' if sub else ""
    return f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{v}</div>{s}</div>'


def page(title, current, lede, body, generated_from, description=""):
    """lede is trusted HTML written in this repo; everything data-driven is escaped before it reaches here."""
    nav = "".join(f'<a href="{h}"{" class=on" if h == current else ""}>{esc(t)}</a>' for h, t in NAV)
    stamp = generated_from.get("built_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    src = "; ".join(f"{esc(k)}: {esc(v)}" for k, v in generated_from.items() if k != "built_at")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><meta name="description" content="{esc(description or 'Weekly7 public explanation page')}">
<style>{CSS}</style></head><body><main>
<nav>{nav}</nav>
<h1>{esc(title)}</h1><div class="sub">{lede}</div>
{body}
<p class="note">Built {esc(stamp)}. Sources: {src or "none found"}. Generated by scripts/site_build.py; sealed test windows and credentials are never read.
Paper trading only; nothing here is investment advice.</p>
</main><script>{JS}</script></body></html>
"""
