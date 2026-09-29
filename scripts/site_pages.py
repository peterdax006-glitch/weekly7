"""Bible Phase 29 - the four generated public pages: Pattern Explorer, Sensitivity, Checklist and unproven, Runs and registry.

Each `build_*` takes the loaded structures from site_sources and returns (html, summary). The summary is the machine
readable digest that site_build stores in the manifest, so a test (or a reader) can see what each page claims without
parsing HTML. Design promises from the Bible: negative experiments are never hidden (they get their own visible section
and colour), every claim shows its sample size and uncertainty, and unproven things are listed as unproven."""
from scripts import site_charts as C
from scripts.site_render import (badge, cell, controls, esc, fmt_int, fmt_num, fmt_pct, fmt_t, page, sign_cls, table, tile)
from scripts.site_sources import LIVE_STATES, STATUS_ORDER, era_breakdown, era_of, provenance_coverage, registry_view

STATUS_KIND = {"active": "b-good", "rescoped": "b-good", "watch": "b-warn", "discarded": "b-bad", "rejected": "b-bad",
               "no_gain": "", "duplicate": "", "candidate": "", "failed": "b-bad", "cause_search": "b-warn"}
STATUS_COLOUR = {"active": "var(--good)", "rescoped": "var(--s1)", "watch": "var(--warn)", "discarded": "var(--bad)", "rejected": "var(--muted)",
                 "no_gain": "var(--s2)", "duplicate": "var(--grid)"}
STATUS_WORDS = {
    "active": "passed discovery, confirmation and false-discovery control and is still working recently",
    "rescoped": "stopped working overall but still holds inside one market context, so it is allowed only there",
    "watch": "flagged as weakening; kept but not traded",
    "discarded": "worked once, then failed recently with no context in which it still holds; kept on record, never traded",
    "rejected": "did not pass the P(real) bar; recorded, never used",
    "no_gain": "passed the tests but added nothing to out-of-sample prediction, so it is not used",
    "duplicate": "fires on nearly the same rows as a stronger pattern",
}


def _problems(problems):
    if not problems:
        return ""
    li = "".join(f"<li>{esc(p)}</li>" for p in problems)
    return f'<div class="card"><h2>Data problems this build</h2><ul>{li}</ul><div class="note">Shown so a missing source is never mistaken for a clean result.</div></div>'


# ================================================================ Pattern Explorer
def _history_html(p):
    if not p.get("history"):
        return '<div class="note">No dated lifecycle history stored for this run; the status above is the miner\'s verdict on that run.</div>'
    rows = "".join(f'<div><span class="mono">{esc(h["as_of"])}</span> {esc(h["frm"] or "new")} → <b>{esc(h["to"])}</b> {"(note)" if h["kind"] == "note" else ""} '
                   f'<span class="muted">{esc(h["reason"])}</span></div>' for h in p["history"][-12:])
    return f'<div class="hist">{rows}</div>'


def _evidence_html(p):
    bits = []
    if p.get("t_disc") is not None:
        bits.append(f"discovery half: mean {fmt_pct(p.get('m_disc'), 2, True)}/week, t {fmt_t(p['t_disc'])}")
    if p.get("t_conf") is not None:
        bits.append(f"confirmation half (later data): mean {fmt_pct(p.get('m_conf'), 2, True)}/week, t {fmt_t(p['t_conf'])}")
    if p.get("p_halluc") is not None:
        bits.append(f"chance it is a hallucination (matches the null pool): {fmt_pct(p['p_halluc'], 1)}")
    if p.get("p_coinc") is not None:
        bits.append(f"chance of coincidence after testing thousands of candidates: {fmt_pct(p['p_coinc'], 1)}")
    if p.get("fdr_pass") is not None:
        bits.append("passed false-discovery control" if p["fdr_pass"] else "failed false-discovery control")
    if p.get("duplicate_of"):
        bits.append(f"duplicate of {esc(p['duplicate_of'])}")
    if p.get("discard_reason"):
        bits.append(f"discard reason: {esc(p['discard_reason'])}")
    return "<br>".join(bits) if bits else '<span class="muted">no evidence fields recorded</span>'


def _pattern_row(p):
    st = p["status"]
    expr = (f'<div><b>{esc(p["words"])}</b></div><div class="mono muted">{esc(p["expr"])}</div>'
            f'<details><summary>lifecycle and evidence</summary><div class="note">{esc(STATUS_WORDS.get(st, ""))}.</div>'
            f'{_history_html(p)}<div class="note">{_evidence_html(p)}</div></details>')
    recent = (f'{fmt_pct(p.get("m_recent"), 2, True)}<div class="note">t {fmt_t(p.get("t_recent"))}, n {fmt_int(p.get("n_recent"))}</div>'
              if p.get("m_recent") is not None else "—")
    hist = (f'{fmt_pct(p.get("m_all"), 2, True)}<div class="note">t {fmt_t(p.get("t_all"))}</div>' if p.get("m_all") is not None else "—")
    real = p.get("p_real")
    rcls = "good" if (real or 0) >= 0.9 else ("bad" if real is not None and real < 0.5 else "")
    trust = f'<div class="note">trust {fmt_num(p["trust"], 2)}</div>' if p.get("trust") is not None else ""
    return [
        cell(expr), cell(badge(st, STATUS_KIND.get(st, "")), sort=STATUS_ORDER.index(st) if st in STATUS_ORDER else 99),
        cell(fmt_pct(p.get("effect"), 3, True), sort=p.get("effect"), cls=sign_cls(p.get("effect"))),
        cell(fmt_pct(real, 1) + trust, sort=real, cls=rcls), cell(fmt_int(p.get("n_eff")), sort=p.get("n_eff")),
        cell(f'{fmt_t(p.get("t_disc"))} / {fmt_t(p.get("t_conf"))}', sort=p.get("t_conf")),
        cell(recent, sort=p.get("m_recent")), cell(hist, sort=p.get("m_all")),
        cell(esc(p.get("scope") or "everywhere")),
    ]


def build_explorer(runs, bank, calibration, problems, generated_from):
    all_p = [p for r in runs for p in r["patterns"]]
    if bank.get("available"):
        all_p = list(bank["patterns"]) + all_p
    counts = {}
    for r in runs:
        for k, v in r["counts"].items():
            counts[k] = counts.get(k, 0) + v
    if bank.get("available"):
        counts = {}
        for p in bank["patterns"]:
            counts[p["status"]] = counts.get(p["status"], 0) + 1
    tested = sum(r["tested"] for r in runs)
    live_n = sum(counts.get(k, 0) for k in LIVE_STATES)

    tiles = "".join([
        tile("Candidates tested", fmt_int(tested), f"across {len(runs)} mining runs"),
        tile("Active or rescoped", f'<span class="{"good" if live_n else "bad"}">{live_n}</span>', "allowed to influence picks"),
        tile("Discarded", fmt_int(counts.get("discarded", 0)), "worked, then failed; kept on record"),
        tile("Rejected", fmt_int(counts.get("rejected", 0)), "did not reach P(real) 0.8"),
        tile("Bank versions", fmt_int(bank.get("version")) if bank.get("available") else "none", "long-term hash-chained bank"),
    ])

    # per-run funnel
    frows = []
    for r in runs:
        t = r["test"]
        bar = C.stacked_bar([(k, r["counts"].get(k, 0), STATUS_COLOUR.get(k, "var(--muted)")) for k in STATUS_ORDER])
        sep = (t.get("real_t_95pct") or 0) - (t.get("null_t_95pct") or 0) if t.get("real_t_95pct") is not None and t.get("null_t_95pct") is not None else None
        frows.append([cell(f'<b>{esc(r["cut"])}</b> <span class="muted">{esc(r["tag"])}</span>'), cell(fmt_int(r["tested"]), sort=r["tested"]),
                      cell(fmt_int(t.get("fdr_pass")), sort=t.get("fdr_pass")),
                      cell(fmt_int(sum(r["counts"].get(k, 0) for k in LIVE_STATES)), sort=sum(r["counts"].get(k, 0) for k in LIVE_STATES)),
                      cell(f'{fmt_num(t.get("null_t_95pct"), 2)} vs {fmt_num(t.get("real_t_95pct"), 2)}'
                           + (f'<div class="note">separation {sep:+.2f}</div>' if sep is not None else ""), sort=sep),
                      cell(fmt_num(t.get("ic_mean"), 4) + f'<div class="note">t {fmt_num(t.get("ic_t"), 2)}, {fmt_int(t.get("weeks"))} weeks</div>', sort=t.get("ic_mean")),
                      cell(bar)])
    funnel = table([("Run (cut date)", True), ("Tested", True), ("Passed FDR", True), ("Live", True), ("Null t95 vs real t95", True),
                    ("Rank IC", True), ("Where the candidates went", False)], frows, "funnel")

    # calibration of P(real)
    cal_html = '<p class="muted">Planted-pattern calibration not found.</p>'
    cal_rows = []
    if calibration:
        bins = []
        for k, v in (calibration.get("calibration") or {}).items():
            try:
                lo, hi = [float(x) for x in k.strip("([]) ").split(",")]
            except ValueError:
                continue
            bins.append(((lo + hi) / 2 if hi <= 1 else hi, v.get("share_truly_real"), v.get("n") or 0))
            cal_rows.append([cell(esc(k)), cell(fmt_int(v.get("n")), sort=v.get("n")), cell(fmt_pct(v.get("share_truly_real"), 1), sort=v.get("share_truly_real"),
                             cls="bad" if (v.get("share_truly_real") or 1) < hi - 0.15 else "")])
        rec = calibration.get("recovery") or {}
        rec_rows = [[cell(esc(k)), cell(fmt_pct(v, 0), sort=v, cls="good" if v >= 0.9 else "bad")] for k, v in rec.items()]
        cal_html = (f'<div class="two"><div>{C.calibration_dots(bins)}<div class="note">A dot on the dashed line means P(real) can be taken at face value. '
                    f'The 0.8–0.9 bin sits well below the line: P(real) is <b>over-confident there</b> (recorded, open).</div>'
                    f'{table([("Claimed P(real)", False), ("Patterns", False), ("Truly real", False)], cal_rows, "cal", sortable=False)}</div>'
                    f'<div>{table([("Planted effect", False), ("Recovered", False)], rec_rows, "rec", sortable=False)}'
                    f'<div class="note">Synthetic panels with effects planted at known sizes; {esc(calibration.get("seeds"))} seeds. '
                    f'False active patterns per run: {fmt_num(calibration.get("false_active_per_run"), 3)} of {fmt_num(calibration.get("active_per_run"), 1)} active '
                    f'({fmt_pct(calibration.get("false_share_of_active"), 1)}).</div></div></div>')

    # table of patterns
    ordered = sorted(all_p, key=lambda p: (STATUS_ORDER.index(p["status"]) if p["status"] in STATUS_ORDER else 99, -(p.get("p_real") or 0), -abs(p.get("t_all") or 0)))
    rows = [_pattern_row(p) for p in ordered]
    ra = [{"status": p["status"], "run": p["run"]} for p in ordered]
    run_names = sorted({p["run"] for p in all_p})
    stat_names = [s for s in STATUS_ORDER if any(p["status"] == s for p in all_p)]
    ptable = table([("Pattern (expression, lifecycle, evidence)", False), ("Status", True), ("Effect / week", True), ("P(real)", True), ("Sample (n eff)", True),
                    ("t discovery / confirmation", True), ("Recent", True), ("Historical", True), ("Context", False)], rows, "patterns", row_attrs=ra)

    live_note = ("" if live_n else '<div class="card"><h2>No pattern is currently live</h2><div class="sub">Every candidate tried so far was rejected or discarded. '
                 "That is the finding, not a page error: the miner requires P(real) ≥ 0.8 and a positive confirmation before a pattern may influence a pick, and none has kept "
                 "clearing that bar on later data. Rejected patterns stay listed below.</div></div>")
    listed = len(all_p)
    total_all = sum(counts.values())
    body = "".join([
        f'<section class="tiles">{tiles}</section>', _problems(problems), live_note,
        f'<section class="card"><h2>What happened to every candidate</h2><div class="sub">Thousands of possible patterns are tried per run. The expected '
        f'number that look good by luck alone is measured with a shuffled null pool (right-hand column of t-values).</div>{funnel}</section>',
        f'<section class="card"><h2>Can P(real) be trusted?</h2>{cal_html}</section>',
        f'<section class="card"><h2>Patterns</h2><div class="sub">Sorted by lifecycle status, then P(real). Recent = the latest 20% of the history; historical = all of it. '
        f'Effect is the weekly return difference shrunk by sample size. {listed:,} of {total_all:,} patterns are listed: the strongest per status and run, so live, failed and rejected ones all appear. '
        f'Full counts are in the table above.</div>'
        + controls("patterns", [("status", "Status", stat_names), ("run", "Run", run_names)], "Search expressions")
        + ptable + "</section>"])
    lede = ("Every pattern the system has tested, including the ones that failed. A pattern is a rule such as "
            "<i>indicator A in its top fifth and indicator B in its bottom fifth</i> that preceded a better-than-average next week.")
    summary = {"patterns_listed": len(all_p), "live": live_n, "runs": len(runs), "tested": tested, "status_counts": counts,
               "listed": listed}
    return page("Pattern Explorer", "explorer.html", lede, body, generated_from, "Every tested trading pattern with P(real), evidence and lifecycle."), summary


# ================================================================ Sensitivity
def _knob_card(k, crit, idx=0):
    vals = k["values"]
    rows = []
    for v in vals:
        kind = "selected" if v["is_selected"] else ("good" if (v["effect"] or 0) > 0 and v["beats_multiple_testing"] else ("bad" if (v["effect"] or 0) < 0 else "flat"))
        rows.append({"label": f'{v["value"]}' + ("  (selected)" if v["is_selected"] else ""), "est": v["effect"], "lo": v["lo"], "hi": v["hi"], "kind": kind})
    tr = []
    for v in vals:
        tr.append([cell(esc(v["value"]) + (" " + badge("selected", "b-good") if v["is_selected"] else "")),
                   cell(fmt_num(v["effect"], 5), sort=v["effect"], cls=sign_cls(v["effect"])),
                   cell(f'{fmt_num(v["lo"], 5)} to {fmt_num(v["hi"], 5)}', sort=v["lo"]),
                   cell(fmt_t(v["t"]) + (" " + badge("survives correction", "b-good") if v["beats_multiple_testing"] else ""), sort=v["t"]),
                   cell(fmt_pct(v["same_way"], 0), sort=v["same_way"]), cell(C.sign_bars(v["per_year"], 120, 22)),
                   cell(esc(v.get("consistency") or "—") + f' <span class="muted">({esc(v.get("class") or "")})</span>'),
                   cell(fmt_num(v["d_worst_dd"], 3), sort=v["d_worst_dd"], cls=sign_cls(v["d_worst_dd"]))])
    inner = (C.interval_plot(rows, fmt=lambda x: f"{x:+.5f}") +
             table([("Tested value", False), ("Effect: change in mean weekly return", True), ("95% interval (across years)", False), ("t", True),
                    ("Years same direction", True), ("Year by year", False), ("Stability", False), ("Change in worst drawdown", True)], tr, f"k_{idx}"))
    neg = f'<span class="bad">{k["negative"]} worse</span>' if k["negative"] else "0 worse"
    head = (f'<summary><b>{esc(k["knob"])}</b> <span class="muted">({esc(k["group"])})</span> — selected <b>{esc(k["selected"])}</b>, '
            f'{k["positive"]} better, {neg}, sensitivity {esc(k.get("class") or "n/a")}</summary>')
    return f'<details class="card" data-group="{esc(k["group"])}">{head}<div class="sub">Reason: {esc(k["reason"])}</div>{inner}</details>'


def build_sensitivity2(sens, problems, generated_from):
    knobs = sens["knobs"]
    crit = sens["bonferroni_t"]
    worse = [(k, v) for k in knobs for v in k["values"] if (v["effect"] or 0) < 0]
    survive = [(k, v) for k in knobs for v in k["values"] if v["beats_multiple_testing"]]
    tiles = "".join([tile("Settings examined", fmt_int(len(knobs))), tile("Values tested", fmt_int(sens["n_variants"]), f"each on {sens['years']} blind years"),
                     tile("Made things worse", f'<span class="bad">{len(worse)}</span>', "shown below, not hidden"),
                     tile("Survive multiple testing", fmt_int(len(survive)), f"|t| ≥ {fmt_num(crit, 2)}" if crit else ""),
                     tile("Champion mean week", fmt_pct(sens.get("champion_mean_week"), 2, True))])
    ranking = [[cell(f'<b>{esc(k["knob"])}</b>'), cell(esc(k["group"])), cell(esc(k["selected"])), cell(fmt_num(k["range"], 5), sort=k["range"]),
                cell(esc(k["class"] or "—")), cell(esc(k["best_value"])), cell(esc(k["reason"]), cls="muted")] for k in knobs]
    rk = table([("Setting", True), ("Kind", True), ("Selected", True), ("Range of effect", True), ("Sensitivity class", True), ("Best tested", True), ("Why the selected value stays", False)],
               ranking, "ranking")
    negs = [[cell(f'{esc(k["knob"])} = {esc(v["value"])}'), cell(fmt_num(v["effect"], 5), sort=v["effect"], cls="bad"), cell(fmt_t(v["t"]), sort=v["t"]),
             cell(fmt_pct(v["same_way"], 0), sort=v["same_way"]), cell(esc(v.get("consistency") or "—"))]
            for k, v in sorted(worse, key=lambda kv: kv[1]["effect"])[:60]]
    neg_tab = table([("Alternative tried", True), ("Effect / week", True), ("t", True), ("Years same direction", True), ("Stability", False)], negs, "negatives")
    cards = "".join(_knob_card(k, crit, i) for i, k in enumerate(knobs))
    ch = sens.get("champion") or {}
    champ_rows = [[cell(esc(k)), cell(esc("off" if v is None else v))] for k, v in ch.items() if k != "ew"]
    ew = ch.get("ew") or {}
    ew_rows = [[cell(esc(k)), cell(fmt_num(v, 2), sort=v, cls=sign_cls(v))] for k, v in sorted(ew.items(), key=lambda kv: -abs(kv[1]))]
    body = "".join([
        f'<section class="tiles">{tiles}</section>', _problems(problems),
        f'<section class="card"><h2>How to read this</h2><div class="sub">Each setting is changed alone while everything else stays at the champion. '
        f'<b>Effect</b> is the change in average weekly return over the champion; the <b>interval</b> comes from the year-by-year effects; <b>stability</b> is the share of years the change '
        f'pointed the same way. With {sens["n_variants"]} values tried, some will look good by luck, so a single value only counts when |t| ≥ {fmt_num(crit, 2)} (Bonferroni). '
        f'Alternatives that lost are listed in full.</div></section>',
        f'<section class="card"><h2>Settings ranked by how much they matter</h2>{controls("ranking", placeholder="Search settings")}{rk}</section>',
        f'<section class="card"><h2>Negative results (never hidden)</h2>{controls("negatives", placeholder="Search")}{neg_tab}</section>',
        f'<section class="card"><h2>Every setting in detail</h2><div class="note">Open a setting to see each tested value, its interval and its year-by-year sign pattern.</div>{cards}</section>',
        f'<div class="two"><section class="card"><h2>Champion configuration</h2>{table([("Setting", False), ("Value", False)], champ_rows, "champ", sortable=False)}</section>'
        f'<section class="card"><h2>Indicator weights</h2>{table([("Indicator", False), ("Weight", False)], ew_rows, "ew", sortable=False)}</section></div>'])
    lede = ("Which settings of the trading rules were tested, how big each effect was, how sure we are, and why the chosen value was kept. "
            "Values that hurt are shown as clearly as values that helped.")
    summary = {"settings": len(knobs), "variants": sens["n_variants"], "worse": len(worse), "survive_correction": len(survive), "bonferroni_t": crit}
    return page("Sensitivity", "sensitivity2.html", lede, body, generated_from, "Tested parameter values, effect, uncertainty, stability and reason selected."), summary


# ================================================================ Checklist and unproven
def build_checklist(ck, unproven, problems, generated_from):
    counts = ck["counts"]
    total = ck["total"] or 1
    colours = {"x": "var(--good)", "~": "var(--s1)", "?": "var(--warn)", "!": "var(--bad)", " ": "var(--grid)"}
    names = {"x": "validated", "~": "implemented / testing", "?": "unproven", "!": "failed", " ": "not started"}
    tiles = "".join(tile(names[s].capitalize(), fmt_int(counts.get(s, 0)), fmt_pct(counts.get(s, 0) / total, 0)) for s in ("x", "~", "?", "!", " "))
    overall = C.stacked_bar([(names[s], counts.get(s, 0), colours[s]) for s in ("x", "~", "?", "!", " ")])
    prow = []
    for p in ck["phases"]:
        c = {}
        for it in p["items"]:
            c[it["state"]] = c.get(it["state"], 0) + 1
        prow.append([cell(f'<b>{esc(p["name"].split(":")[0])}</b>'), cell(fmt_int(len(p["items"])), sort=len(p["items"])),
                     cell(fmt_int(c.get("x", 0)), sort=c.get("x", 0)),
                     cell(C.stacked_bar([(names[s], c.get(s, 0), colours[s]) for s in ("x", "~", "?", "!", " ")]))])
    ptab = table([("Area", True), ("Items", True), ("Validated", True), ("Progress", False)], prow, "phases")
    irows, ra = [], []
    for p in ck["phases"]:
        for it in p["items"]:
            kind = {"x": "b-good", "!": "b-bad", "?": "b-warn", "~": "", " ": ""}[it["state"]]
            ev = esc(it["evidence"]) if it["evidence"] else '<span class="muted">none recorded</span>'
            irows.append([cell(f'<span class="mono">{esc(it["id"])}</span>'), cell(badge(it["state_name"], kind)),
                          cell(f'<b>{esc(it["title"])}</b>' + (f'<div class="note">{esc(it["detail"])}</div>' if it["detail"] else "")), cell(ev)])
            ra.append({"state": it["state_name"], "phase": p["name"].split(":")[0]})
    itab = table([("ID", True), ("State", True), ("Item", False), ("Evidence", False)], irows, "items", row_attrs=ra)
    urows = [[cell(f'<span class="mono">{esc(u["id"])}</span>'), cell(esc(u["phase"])), cell(badge(u["state_name"], {"!": "b-bad", "?": "b-warn"}.get(u["state"], ""))),
              cell(f'<b>{esc(u["title"])}</b><div class="note">{esc(u["detail"] or u["raw"])}</div>')] for u in unproven["items"]]
    utab = table([("ID", True), ("Area", True), ("State", True), ("What is unproven and why", False)], urows, "unproven")
    nrows = [[cell(esc(n["t"])), cell(badge(n["event"], "b-bad")), cell(esc(n["id"] or "")), cell(esc(n["text"]))] for n in unproven["negatives"]]
    ntab = table([("When", True), ("Event", True), ("Id", False), ("Reason", False)], nrows, "neg")
    warn = f'<div class="note">{ck["unparsed"]} checklist line(s) did not match the expected format and are not shown.</div>' if ck["unparsed"] else ""
    body = "".join([
        f'<section class="tiles">{tiles}</section>', _problems(problems),
        f'<section class="card"><h2>Overall</h2>{overall}<div class="note">Nothing is marked validated merely because code exists; a box turns green only with evidence written next to it.</div></section>',
        f'<section class="card"><h2>Progress by area</h2>{ptab}</section>',
        f'<section class="card"><h2>UNPROVEN: what we do not yet know</h2><div class="sub">Every item that is not validated, failed first.</div>{controls("unproven", placeholder="Search unproven items")}{utab}</section>',
        f'<section class="card"><h2>Rollbacks and voided results</h2><div class="sub">Changes that were adopted and then undone, or results thrown away because the check failed.</div>{ntab}</section>',
        f'<section class="card"><h2>Full checklist</h2>{controls("items", [("state", "State", sorted({i["state_name"] for p in ck["phases"] for i in p["items"]})), ("phase", "Area", [p["name"].split(":")[0] for p in ck["phases"]])], "Search checklist")}{itab}{warn}</section>'])
    lede = "Live status of the master checklist (state/CHECKLIST.md) and an honest list of what is not proven yet."
    summary = {"items": ck["total"], "counts": counts, "unproven": len(unproven["items"]), "negatives": len(unproven["negatives"]), "unparsed": ck["unparsed"]}
    return page("Checklist and unproven", "checklist.html", lede, body, generated_from, "Live checklist status and the unproven list."), summary


# ================================================================ Runs and registry
def _reg_row(r):
    v = registry_view(r)
    head = ", ".join(f"{k}={_short(x)}" for k, x in list(v["headline"].items())[:7])
    nested = "; ".join(f"{k}: {x}" for k, x in v["nested"].items())
    prov = " ".join(f'<span class="badge">{esc(k)} {esc(str(x)[:10])}</span>' for k, x in v["prov"].items()) or '<span class="muted">no provenance recorded</span>'
    det = f'<details><summary>record</summary><pre>{esc(v["detail"])}</pre></details>'
    return [cell(esc(r.get("t") or ""), sort=r.get("t")), cell(badge(r.get("event") or "?")), cell(f'<span class="mono">{esc(r.get("experiment_id") or "")}</span>'),
            cell(esc(head) + (f'<div class="note">{esc(nested)}</div>' if nested else "") + det), cell(prov)]


def _short(x):
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)[:40]


def build_runs(cycles, registry, reports, problems, generated_from):
    cs = cycles["cycles"]
    eras = era_breakdown(cs)
    mw = [c.get("mean_week") for c in cs]
    beat = sum(1 for c in cs if (c.get("year_return") or 0) > (c.get("market_year_return") or 0))
    ge7 = sum((c.get("weeks_ge_7") or 0) for c in cs)
    tiles = "".join([tile("Revealed windows", fmt_int(len(cs)), f"{cycles['hidden']} still sealed and not shown"),
                     tile("Beat the market", f"{beat} of {len(cs)}" if cs else "—"), tile("Weeks at +7% or better", fmt_int(ge7)),
                     tile("Registry records", fmt_int(len(registry))), tile("Report files", fmt_int(len(reports)))])
    crow, ra = [], []
    for c in sorted(cs, key=lambda c: str(c["year"])):
        yr, mk = c.get("year_return"), c.get("market_year_return")
        crow.append([cell(f'<span class="mono">{esc(c["run_id"])}</span>'), cell(esc(c["year"]), sort=c["year"]), cell(esc(era_of(c["year"]))),
                     cell(fmt_pct(c.get("mean_week"), 2, True), sort=c.get("mean_week"), cls=sign_cls(c.get("mean_week"))),
                     cell(fmt_int(c.get("weeks_ge_7")), sort=c.get("weeks_ge_7")), cell(fmt_int(c.get("weeks_le_m7")), sort=c.get("weeks_le_m7"), cls="bad" if c.get("weeks_le_m7") else ""),
                     cell(fmt_pct(yr, 1, True), sort=yr, cls=sign_cls(yr)), cell(fmt_pct(mk, 1, True), sort=mk),
                     cell(badge("beat", "b-good") if (yr or 0) > (mk or 0) else badge("lost", "b-bad")),
                     cell(fmt_pct(c.get("max_dd"), 1), sort=c.get("max_dd"), cls="bad"), cell(fmt_int(c.get("brakes")), sort=c.get("brakes"))])
        ra.append({"era": era_of(c["year"])})
    ctab = table([("Run", True), ("Year", True), ("Era", True), ("Mean week", True), ("Weeks ≥ +7%", True), ("Weeks ≤ −7%", True), ("Year return", True),
                  ("Market", True), ("vs market", False), ("Max drawdown", True), ("Brakes", True)], crow, "cycles", row_attrs=ra)
    erow = [[cell(f'<b>{esc(e["era"])}</b>'), cell(fmt_int(e["windows"]), sort=e["windows"]), cell(fmt_pct(e["mean_week"], 2, True), sort=e["mean_week"], cls=sign_cls(e["mean_week"])),
             cell(fmt_pct(e["beat_market"], 0), sort=e["beat_market"]), cell(fmt_num(e["ge7"], 2), sort=e["ge7"]), cell(fmt_pct(e["worst_dd"], 1), sort=e["worst_dd"], cls="bad")] for e in eras]
    etab = table([("Era", True), ("Windows", True), ("Mean weekly return", True), ("Beat market", True), ("Weeks ≥ +7% per window", True), ("Worst drawdown", True)], erow, "eras")
    rd = [[cell(fmt_int(r["round"]), sort=r["round"]), cell(badge("adopted", "b-good") if r["adopted"] else badge("not adopted")),
           cell(fmt_pct(r["before"], 3, True), sort=r["before"]), cell(fmt_pct(r["after"], 3, True), sort=r["after"]),
           cell((badge("voided", "b-bad") + " " + esc(r["voided"])) if r["voided"] else "")] for r in cycles["rounds"]]
    rtab = table([("Round", True), ("Adjustment", False), ("Avg week before", True), ("Avg week after", True), ("Note", False)], rd, "rounds")
    events = sorted({r.get("event") or "?" for r in registry})
    reg = [_reg_row(r) for r in sorted(registry, key=lambda r: r.get("t") or "", reverse=True)]
    rra = [{"event": r.get("event") or "?"} for r in sorted(registry, key=lambda r: r.get("t") or "", reverse=True)]
    regtab = table([("When", True), ("Event", True), ("Experiment id", False), ("Result", False), ("Provenance", False)], reg, "registry", row_attrs=rra)
    cov = provenance_coverage(registry)
    covtab = table([("Provenance field", False), ("Records that carry it", False)],
                   [[cell(f'<span class="mono">{esc(k)}</span>'), cell(fmt_pct(v, 0) + (' <span class="bad">(early records predate it)</span>' if v < 0.5 else ""), sort=v)] for k, v in cov.items()],
                   "cov", sortable=False)
    counts = {}
    for r in registry:
        counts[r.get("event") or "?"] = counts.get(r.get("event") or "?", 0) + 1
    evrows = [[cell(esc(k)), cell(fmt_int(v), sort=v)] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    evtab = table([("Event", True), ("Records", True)], evrows, "evcount")
    rep = [[cell(esc(r["area"])), cell(f'<span class="mono">{esc(r["name"])}</span>'), cell(esc(r["kind"])), cell(fmt_int(r["bytes"] / 1024) + " KB", sort=r["bytes"]),
            cell(esc(", ".join(f"{k}={_short(v)}" for k, v in r["head"].items())), cls="muted")] for r in reports]
    reptab = table([("Area", True), ("File", True), ("Type", True), ("Size", True), ("Headline numbers", False)], rep, "reports")
    body = "".join([
        f'<section class="tiles">{tiles}</section>', _problems(problems),
        f'<section class="card"><h2>Blind simulated years</h2><div class="sub">Each row is one 12-month window that was tested blind and has since been revealed. Windows still sealed are not listed, '
        f'and their contents are never read by this site. Mean-week trend across windows: {C.sparkline(mw)}</div>{controls("cycles", [("era", "Era", [e["era"] for e in eras])], "Search runs")}{ctab}</section>',
        f'<section class="card"><h2>By era</h2>{etab}<div class="note">Eras with few windows say little; the count is shown for that reason.</div></section>',
        f'<section class="card"><h2>Weekly self-adjustment rounds</h2>{rtab}</section>',
        f'<section class="card"><h2>Experiment registry</h2><div class="sub">Append-only log of every experiment, including failures. Sort by column or filter by event.</div>'
        f'<div class="two"><div>{evtab}</div><div>{covtab}</div></div>{controls("registry", [("event", "Event", events)], "Search experiments")}{regtab}</section>',
        f'<section class="card"><h2>Report files</h2><div class="sub">Artefacts produced by research and simulation runs (listing only).</div>{controls("reports", placeholder="Search reports")}{reptab}</section>'])
    lede = "Every blind simulated year that has been revealed, the weekly self-adjustments, and the append-only registry of experiments."
    summary = {"windows": len(cs), "hidden_windows_not_shown": cycles["hidden"], "registry_records": len(registry), "events": counts, "reports": len(reports)}
    return page("Runs and registry", "runs.html", lede, body, generated_from, "Revealed simulation runs, per-era results and the experiment registry."), summary
