"""Learning reports K01-K15, the J15 research-priority dashboard data and the section-46 health dashboard data
(contract C62 sections 46-48, 59, 65, 79, 51; checklist K01-K15, J11, J13-J15).

STATUS: IMPLEMENTED - NOT VALIDATED (synthetic-store unit tests only; no real archive was read).

Every report is a generator over ARTEFACTS: plain JSON / JSONL files written by the other learning modules (module stores, the
scorecard store, the master checklist). Rules that make a report an instrument and not a press release:

  * numbers are computed from the artefacts here, never typed (prose goes stale);
  * a missing, empty or unreadable artefact makes that report - or that section - say UNMEASURED with the reason: never blank, never
    a success, never a zero;
  * every report states its section-59 label honestly (IMPLEMENTED - NOT VALIDATED / FAILED VALIDATION / INSUFFICIENT EVIDENCE);
    VALIDATED appears only when the master checklist holds that item as VALIDATED AND its evidence file exists on disk;
  * "learning improved" is refused unless scorecard.gate_improvement_claim passes on the newest stored scorecard (controls A-E, unseen
    year, memorisation and leak gates); a refused claim is listed in the report as refused, with the blockers;
  * nothing dated at or after `now` is read (rows are dropped and counted, or FirewallBreach in strict mode);
  * rendered markdown passes scorecard.assert_claim_ok: a report cannot contain a word its own evidence forbids.

Reports are frozen dataclasses with a content id; the JSON and the markdown are two renderings of the same record."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .core import (Epistemic, FailureCause, FirewallBreach, Health, Lifecycle, Promotion, Provenance, ValidationLabel, as_date,
                   canonical_json, current_code_hash, stable_hash)
from . import scorecard as SC

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = ROOT / "state" / "research" / "learning"
DEFAULT_OUT = ROOT / "state" / "research" / "learning_reports"
DEFAULT_CHECKLIST = ROOT / "state" / "build" / "SELF_LEARNING_MASTER_CHECKLIST.json"

ARTEFACTS = {
    "scorecards": "scorecards.jsonl",            # scorecard.ScorecardStore (hash-chained)
    "curve": "curve.json",                       # list of learning_curve.CurvePoint records
    "transfer": "transfer.json",                 # transfer.TransferReport.as_record() (one, or a list of them)
    "identity": "identity.json",                 # identity_firewall.IdentityReport as {verdicts:[...], base_ic, deterministic}
    "knowledge": "knowledge.jsonl",              # knowledge.KnowledgeStore.dump (every version)
    "failures": "failures.jsonl",                # failure.Classification.to_dict() rows
    "retirement": "retirement.jsonl",            # retirement.Transition rows
    "contradictions": "contradictions.jsonl",    # contradiction.LedgerEntry rows
    "missed": "missed_winners.json",             # {reasons, distinctions, walk_forward}
    "experiments": "experiments.jsonl",          # experiment_memory.ExperimentRecord dicts
    "queue": "research_queue.json",              # research queue items (list)
    "meta": "meta_learning.json",                # meta_learning.MetaStore lists by kind
    "promotions": "promotions.jsonl",            # promotion.PromotionDecision.to_dict() rows
    "delta": "learning_delta.json",              # section-65 learning delta by dimension
}
DATE_KEYS = ("at", "recorded_at", "as_of", "now", "enqueued_at", "updated_at", "created_at", "resolved_at", "when", "date")
DELTA_DIMS = ("movement_delta", "selection_delta", "direction_delta", "risk_delta", "drawdown_delta", "band_share_delta",
              "transfer_delta", "calibration_delta")
UNMEASURED = "UNMEASURED"
LABEL_ORDER = {ValidationLabel.FAILED_VALIDATION: 3, ValidationLabel.INSUFFICIENT_EVIDENCE: 2, ValidationLabel.NOT_VALIDATED: 1,
               ValidationLabel.VALIDATED: 0}
_CLAIM_WORDS = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(SC.FORBIDDEN_ALWAYS + SC.FORBIDDEN_WITHOUT_GATE, key=len, reverse=True)) + r")\b",
                          re.IGNORECASE)


# ---------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------
def num(x, default: float = float("nan")) -> float:
    """JSON number -> float. None, '', 'nan' and non-numbers are NaN (not measured), never zero."""
    if x is None or isinstance(x, bool):
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if not math.isnan(v) else default


def fin(x) -> bool:
    return math.isfinite(num(x))


def fmt(x, nd: int = 4, signed: bool = False) -> str:
    v = num(x)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def defang(text: Any) -> str:
    """Artefact text may contain words a report may not assert ('improved', 'done'...). Quoted data is shown with them replaced by '[claim word]',
    so a stored note can never make the report claim something its evidence does not."""
    return _CLAIM_WORDS.sub("[claim word]", str(text))


def jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if hasattr(x, "item") and callable(x.item):
        return jsonable(x.item())
    if hasattr(x, "value") and not isinstance(x, (str, int)):
        return x.value
    if isinstance(x, (dt.date, dt.datetime)):
        return x.isoformat()
    return x


def row_date(row: Mapping) -> dt.date | None:
    for k in DATE_KEYS:
        v = row.get(k)
        if isinstance(v, str) and len(v) >= 10:
            try:
                return as_date(v)
            except ValueError:
                continue
    return None


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def share(k: int, n: int) -> float:
    return k / n if n else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a share; (nan, nan) when n == 0. Reports show shares WITH their uncertainty."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------------------------------------------------------
# artefact access
# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Source:
    name: str
    path: str
    status: str                  # FOUND | MISSING | EMPTY | UNREADABLE
    n: int = 0
    sha: str = ""
    error: str = ""
    excluded_future: int = 0     # rows dated at/after `now`, dropped before any number was computed

    @property
    def usable(self) -> bool:
        return self.status == "FOUND" and self.n > 0


class ReportContext:
    """Where artefacts live, what time it is (`now`), and the claim gate. One context per generation run; artefacts are read once."""

    def __init__(self, root: Path | str = DEFAULT_INPUTS, now=None, *, seed: int = 0, checklist: Path | str | None = DEFAULT_CHECKLIST,
                 paths: Mapping[str, Path | str] | None = None, strict_future: bool = False, code_hash: str | None = None):
        if now is None:
            raise FirewallBreach("ReportContext needs an explicit `now`: a report never reads the wall clock")
        self.root, self.now, self.seed = Path(root), as_date(now), int(seed)
        self.checklist_path = Path(checklist) if checklist else None
        self.paths = {k: Path(v) for k, v in (paths or {}).items()}
        self.strict_future = strict_future
        self.code_hash = code_hash if code_hash is not None else current_code_hash()
        self._raw: dict[str, tuple[Any, Source]] = {}

    def path_of(self, name: str) -> Path:
        return self.paths.get(name) or self.root / ARTEFACTS[name]

    def _read(self, name: str) -> tuple[Any, Source]:
        if name in self._raw:
            return self._raw[name]
        p = self.path_of(name)
        if not p.exists():
            out = (None, Source(name, str(p), "MISSING"))
        else:
            try:
                text = p.read_text(encoding="utf-8")
                if p.suffix == ".jsonl":
                    obj = []
                    for i, line in enumerate(text.splitlines(), 1):
                        if line.strip():
                            try:
                                obj.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                raise ValueError(f"line {i}: {e.msg}") from e
                else:
                    obj = json.loads(text) if text.strip() else None
                out = (obj, Source(name, str(p), "FOUND", sha=sha_of(p)))
            except (OSError, ValueError) as e:
                out = (None, Source(name, str(p), "UNREADABLE", error=str(e)))
        self._raw[name] = out
        return out

    def _past_only(self, rows: list, src: Source) -> tuple[list, Source]:
        keep, dropped = [], 0
        for r in rows:
            d = row_date(r) if isinstance(r, Mapping) else None
            if d is not None and d >= self.now:
                if self.strict_future:
                    raise FirewallBreach(f"{src.name}: record dated {d} is not before now={self.now}")
                dropped += 1
            else:
                keep.append(r)
        status = "FOUND" if keep else "EMPTY"
        return keep, dataclasses.replace(src, status=status, n=len(keep), excluded_future=dropped)

    def rows(self, name: str) -> tuple[list[dict], Source]:
        """The artefact as a list of dict rows dated before `now`. A JSON object holding a list under rows/records/items is unwrapped."""
        obj, src = self._read(name)
        if src.status != "FOUND":
            return [], src
        if isinstance(obj, Mapping):
            for k in ("rows", "records", "items"):
                if isinstance(obj.get(k), list):
                    obj = obj[k]
                    break
            else:
                obj = [obj]
        if not isinstance(obj, list) or not all(isinstance(r, Mapping) for r in obj):
            return [], dataclasses.replace(src, status="UNREADABLE", error="artefact is not a list of records")
        return self._past_only(list(obj), src)

    def obj(self, name: str) -> tuple[dict | None, Source]:
        """A JSON-object artefact (a list is reduced to its newest record dated before `now`). None when it is not there."""
        obj, src = self._read(name)
        if src.status != "FOUND":
            return None, src
        if isinstance(obj, list):
            rows, src2 = self._past_only([r for r in obj if isinstance(r, Mapping)], dataclasses.replace(src, n=len(obj)))
            return (dict(rows[-1]) if rows else None), src2
        if not isinstance(obj, Mapping):
            return None, dataclasses.replace(src, status="UNREADABLE", error="artefact is not an object")
        d = row_date(obj)
        if d is not None and d >= self.now:
            if self.strict_future:
                raise FirewallBreach(f"{name}: record dated {d} is not before now={self.now}")
            return None, dataclasses.replace(src, status="EMPTY", excluded_future=1)
        return dict(obj), dataclasses.replace(src, n=1)

    # ---- scorecards and the claim gate ------------------------------------------------------------------
    def cards(self) -> tuple[list[dict], Source, list[str]]:
        """Stored scorecard records dated before `now`, the source, and the hash-chain problems of the WHOLE file (an edited later line
        still taints the earlier ones: history is one chain)."""
        obj, src = self._read("scorecards")
        if src.status != "FOUND":
            return [], src, []
        recs, src = self._past_only([r.get("record", r) for r in obj if isinstance(r, Mapping)], src)
        return recs, src, SC.ScorecardStore(self.path_of("scorecards")).verify()

    def latest_card(self) -> tuple[SC.LearningScorecard | None, dict | None, list[str]]:
        rows, src, problems = self.cards()
        if not rows:
            return None, None, problems
        try:
            return card_from_record(rows[-1]), rows[-1], problems
        except (KeyError, TypeError, ValueError) as e:
            return None, rows[-1], problems + [f"latest scorecard record cannot be rebuilt: {e}"]

    def decision(self) -> tuple[SC.ClaimDecision | None, SC.LearningScorecard | None, list[str]]:
        """The claim gate on the newest stored scorecard. A store whose hash chain is broken, or a card that cannot be rebuilt, blocks
        every claim (fail closed). No scorecard at all -> None: nothing may be called an improvement."""
        card, rec, problems = self.latest_card()
        if problems:
            chk = SC.GateCheck("store_intact", False, True, "scorecard store integrity: " + "; ".join(problems[:3]))
            return SC.ClaimDecision(False, (chk,), ValidationLabel.FAILED_VALIDATION), card, problems
        if card is None:
            return None, None, problems
        return SC.gate_improvement_claim(card), card, problems


def _measured(d: Any) -> SC.Measured:
    if not isinstance(d, Mapping):
        return SC.Measured.untested("absent from record")
    try:
        st = SC.MStatus(str(d.get("status", "UNTESTED")))
    except ValueError:
        st = SC.MStatus.UNTESTED
    return SC.Measured(num(d.get("value")), num(d.get("lo")), num(d.get("hi")), int(num(d.get("n"), 0)), st, str(d.get("note", "")))


def card_from_record(rec: Mapping) -> SC.LearningScorecard:
    """Rebuild a LearningScorecard from its stored record. Per-unit control gains are not stored (only their hash), so the
    resemblance test of the gate cannot be re-run from a stored card: reports say so in their caveats."""
    kw: dict[str, Any] = {}
    for f in SC.SECTION_47_FIELDS + ("learning_gain",):
        v = rec.get(f)
        if f == "future_leak_status":
            kw[f] = SC.LeakStatus(str(v)) if v else SC.LeakStatus.UNAUDITED
        elif f == "compute_cost":
            cc = v if isinstance(v, Mapping) else {}
            kw[f] = SC.ComputeCost(num(cc.get("cpu_seconds")), num(cc.get("wall_seconds")), num(cc.get("peak_ram_mb")),
                                   int(num(cc.get("n_fits"), 0)), int(num(cc.get("n_evaluations"), 0)))
        else:
            kw[f] = _measured(v)
    controls = {}
    for name, c in (rec.get("controls") or {}).items():
        controls[name] = SC.ControlResult(name, _measured(c.get("gain")), c.get("detected"), (), str(c.get("note", "")))
    return SC.LearningScorecard(learner_version=str(rec["learner_version"]), now=as_date(rec["now"]), code_hash=str(rec.get("code_hash", "")),
                                seed=rec.get("seed"), controls=controls, data_hash=str(rec.get("data_hash", "")),
                                config_hash=str(rec.get("config_hash", "")), controls_hash=str(rec.get("controls_hash", "")),
                                transfer_verdict=str(rec.get("transfer_verdict", "UNTESTED")), curve_verdict=str(rec.get("curve_verdict", "UNTESTED")),
                                label=ValidationLabel.NOT_VALIDATED, **kw)


# ---------------------------------------------------------------------------------------------------------------
# the report record
# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Table:
    title: str
    columns: tuple
    rows: tuple

    def markdown(self) -> str:
        if not self.rows:
            return f"**{defang(self.title)}**: {UNMEASURED} (no rows)"
        head = "| " + " | ".join(defang(c) for c in self.columns) + " |\n|" + "---|" * len(self.columns)
        body = ["| " + " | ".join(defang(c).replace("|", "/") for c in r) + " |" for r in self.rows]
        return f"**{defang(self.title)}**\n\n" + head + "\n" + "\n".join(body)


@dataclasses.dataclass(frozen=True)
class ClaimVerdict:
    text: str
    accepted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class Report:
    key: str
    title: str
    checklist: tuple
    now: str
    code_hash: str
    seed: int
    label: ValidationLabel
    status: str                          # MEASURED | PARTIAL | UNMEASURED
    headline: str
    facts: Mapping
    tables: tuple
    notes: tuple
    unmeasured: tuple                    # reasons a section could not be computed
    sources: tuple
    claim_statement: str
    claims: tuple = ()

    @property
    def measured(self) -> bool:
        return self.status != UNMEASURED

    def to_record(self) -> dict:
        rec = {"key": self.key, "title": self.title, "checklist": list(self.checklist), "now": self.now, "code_hash": self.code_hash,
               "seed": self.seed, "label": self.label.value, "status": self.status, "headline": self.headline, "facts": jsonable(self.facts),
               "tables": [{"title": t.title, "columns": list(t.columns), "rows": [jsonable(list(r)) for r in t.rows]} for t in self.tables],
               "notes": list(self.notes), "unmeasured": list(self.unmeasured), "sources": [jsonable(s) for s in self.sources],
               "claim_statement": self.claim_statement,
               "claims": [{"text": c.text, "accepted": c.accepted, "reason": c.reason} for c in self.claims]}
        rec["report_id"] = stable_hash(rec)
        return rec

    def to_json(self) -> str:
        return json.dumps(json.loads(canonical_json(self.to_record())), indent=1, ensure_ascii=False, sort_keys=True)

    def narrative(self) -> str:
        """Every sentence this report asserts itself (headline, notes, claim statement): the text the claim guard reads."""
        return "\n".join([self.headline, *self.notes, *self.unmeasured])

    def to_markdown(self) -> str:
        L = [f"# {self.key}: {self.title}", "", f"**{self.label.value}**  |  status: {self.status}  |  as of {self.now}  |  code `{self.code_hash}`  |  seed {self.seed}",
             "", self.headline, "", "## Claim status", "", self.claim_statement]
        for c in self.claims:
            L.append(f"- claim {'ACCEPTED' if c.accepted else 'REFUSED'}: \"{defang(c.text)}\" ({c.reason})")
        if self.facts:
            L += ["", "## Numbers (computed from the artefacts)", "", "| quantity | value |", "|---|---|"]
            for k, v in self.facts.items():
                L.append(f"| {k} | {defang(_fact_cell(v))} |")
        for t in self.tables:
            L += ["", t.markdown()]
        if self.notes:
            L += ["", "## Notes"] + [f"- {n}" for n in self.notes]
        if self.unmeasured:
            L += ["", f"## {UNMEASURED}"] + [f"- {u}" for u in self.unmeasured]
        L += ["", "## Sources", "", "| artefact | path | status | rows | sha | dropped as future |", "|---|---|---|---|---|---|"]
        for s in self.sources:
            L.append(f"| {s.name} | {s.path} | {s.status}{': ' + s.error if s.error else ''} | {s.n} | {s.sha} | {s.excluded_future} |")
        L += ["", f"report id `{self.to_record()['report_id']}`"]
        return "\n".join(L)


def _fact_cell(v: Any) -> str:
    if isinstance(v, float):
        return fmt(v, 5)
    if isinstance(v, (list, tuple)):
        return ", ".join(_fact_cell(x) for x in v[:12]) + (f" (+{len(v) - 12})" if len(v) > 12 else "")
    if isinstance(v, Mapping):
        return "; ".join(f"{k}={_fact_cell(x)}" for k, x in list(v.items())[:12])
    return str(v)


def guard_report(rep: Report, decision: SC.ClaimDecision | None) -> None:
    """Refuse a report whose own sentences assert what the evidence does not. With no scorecard the decision is a blocked one."""
    d = decision if decision is not None else SC.ClaimDecision(False, (), ValidationLabel.INSUFFICIENT_EVIDENCE)
    text = rep.narrative()
    if rep.label == ValidationLabel.VALIDATED:
        text = text.replace(ValidationLabel.VALIDATED.value, "")
    SC.assert_claim_ok(text, d)


# ---------------------------------------------------------------------------------------------------------------
# builder shared by every generator
# ---------------------------------------------------------------------------------------------------------------
def checklist_validated(ctx: ReportContext, item_id: str) -> tuple[bool, str]:
    """VALIDATED needs the master checklist to hold `item_id` as VALIDATED with an evidence path that exists (section 59)."""
    p = ctx.checklist_path
    if p is None or not p.exists():
        return False, "no checklist"
    try:
        items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return False, "checklist unreadable"
    for it in items:
        if it.get("id") == item_id:
            ev = [e for e in it.get("evidence", []) if isinstance(e, str)]
            ok = it.get("status") == "VALIDATED" and any((ROOT / e).exists() or Path(e).exists() for e in ev)
            return ok, f"checklist {item_id}: status {it.get('status')}, {len(ev)} evidence entries"
    return False, f"checklist has no item {item_id}"


class Builder:
    def __init__(self, ctx: ReportContext, key: str, title: str, checklist: Sequence[str]):
        self.ctx, self.key, self.title, self.checklist = ctx, key, title, tuple(checklist)
        self.facts: dict[str, Any] = {}
        self.tables: list[Table] = []
        self.notes: list[str] = []
        self.unmeasured: list[str] = []
        self.sources: list[Source] = []

    def source(self, s: Source) -> Source:
        self.sources.append(s)
        return s

    def fact(self, k: str, v: Any) -> None:
        self.facts[k] = v

    def table(self, title: str, columns: Sequence[str], rows: Sequence[Sequence]) -> None:
        self.tables.append(Table(title, tuple(columns), tuple(tuple(r) for r in rows)))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def lost(self, reason: str) -> None:
        self.unmeasured.append(reason)

    def absent(self, s: Source, what: str) -> None:
        self.lost(f"{what}: artefact {s.name} is {s.status}" + (f" ({s.error})" if s.error else "") + (f", {s.excluded_future} rows dated at/after now dropped" if s.excluded_future else ""))

    def build(self, verdict: str | None = None, headline: str = "", *, measured: bool = True, insufficient: bool = False,
              claims: Sequence[str] = ()) -> Report:
        """verdict: 'FAIL' = measured evidence against; 'OK' = nothing against; None = descriptive only. insufficient = too little data."""
        ctx = self.ctx
        decision, card, problems = ctx.decision()
        if not measured:
            status, label = UNMEASURED, ValidationLabel.INSUFFICIENT_EVIDENCE
            headline = headline or f"{UNMEASURED}: " + "; ".join(self.unmeasured[:3])
        else:
            status = "PARTIAL" if self.unmeasured else "MEASURED"
            label = (ValidationLabel.FAILED_VALIDATION if verdict == "FAIL" else ValidationLabel.INSUFFICIENT_EVIDENCE if insufficient
                     else ValidationLabel.NOT_VALIDATED)
            if label == ValidationLabel.NOT_VALIDATED and status == "MEASURED":
                ok, why = checklist_validated(ctx, self.key)
                if ok:
                    label = ValidationLabel.VALIDATED
                    self.note(f"label VALIDATED taken from {why}")
        if not headline.startswith(UNMEASURED) and not measured:
            headline = f"{UNMEASURED}: {headline}"
        if card is not None:
            self.note(f"claim gate read scorecard {card.learner_version} (as of {card.now}); per-unit control gains are not stored, so the "
                      "memoriser-resemblance test cannot be re-run from a stored card")
        if problems:
            self.note("scorecard store problems: " + "; ".join(problems[:3]))
        if decision is None:
            statement = ("No scorecard is stored: the five controls of section 25 have not been run, so there is no claim of improvement. "
                         f"{ValidationLabel.INSUFFICIENT_EVIDENCE.value}.")
        else:
            statement = decision.statement(card) if card is not None else "; ".join(decision.blockers)
        verdicts = tuple(admit_claim(c, decision, card) for c in claims)
        rep = Report(self.key, self.title, self.checklist, ctx.now.isoformat(), ctx.code_hash, ctx.seed, label, status, headline,
                     dict(self.facts), tuple(self.tables), tuple(self.notes), tuple(self.unmeasured), tuple(self.sources), statement, verdicts)
        guard_report(rep, decision)
        return rep

    def none(self, why: str) -> Report:
        return self.build(measured=False, headline=why)


def admit_claim(text: str, decision: SC.ClaimDecision | None, card: SC.LearningScorecard | None) -> ClaimVerdict:
    """A caller's sentence about the learner ('learning improved', 'transfer improved'...) is accepted only if the gate lets it stand.
    Forbidden-always words ('validated', 'done'...) are refused even when the gate passes; field-level claims need that field's controls."""
    d = decision if decision is not None else SC.ClaimDecision(False, (), ValidationLabel.INSUFFICIENT_EVIDENCE)
    bad = SC.claim_violations(text, d)
    if bad:
        why = "; ".join(d.blockers[:3]) if d.blockers else ("no scorecard stored: controls not run" if decision is None else "validation needs a sealed window")
        return ClaimVerdict(text, False, f"words {bad} not supported: {why}")
    if card is not None:
        fields = SC.refuse_unsupported_field_claims(card, text)
        if fields:
            return ClaimVerdict(text, False, "field(s) without their controls: " + ", ".join(fields))
    return ClaimVerdict(text, True, "no unsupported word" if decision is not None else "no claim words")


# ---------------------------------------------------------------------------------------------------------------
# K01 learning curve (section 48)
# ---------------------------------------------------------------------------------------------------------------
def _curve_records(rows: Sequence[Mapping]) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        for k in ("step", "experience_count", "knowledge_count", "validated_knowledge_count"):
            d[k] = int(num(d.get(k), -1))
        for k in ("same_year_gain", "transfer_gain", "risk", "memorization_gap"):
            d[k] = num(d.get(k))
        out.append(d)
    return sorted(out, key=lambda d: d["step"])


def report_k01(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    from . import learning_curve as LC
    b = Builder(ctx, "K01", "Learning curve report", ("K01", "J13"))
    rows, s = ctx.rows("curve")
    b.source(s)
    if not rows:
        b.absent(s, "learning curve")
        return b.none("no curve points, so experience -> future gain cannot be read")
    try:
        curve = LC.curve_from_records("learner", _curve_records(rows))
    except (ValueError, TypeError, KeyError) as e:
        b.lost(f"curve records rejected by learning_curve validation: {e}")
        return b.none("the stored curve is invalid and was not repaired")
    an = LC.analyse_curve(curve, seed=ctx.seed, n_boot=300)
    b.fact("points", len(curve))
    b.fact("experience_first", int(curve.column("experience_count")[0]))
    b.fact("experience_last", int(curve.column("experience_count")[-1]))
    b.fact("verdict", an.verdict.value)
    b.fact("why", defang(an.why))
    b.fact("validated_share_of_knowledge", an.validated_share)
    x, tg = curve.future_improvement()
    b.fact("points_with_measured_transfer_gain", int(len(tg)))
    kc = curve.column("knowledge_count")
    if len(tg) >= 4 and np.std(tg) > 0:
        kx = kc[np.isfinite(curve.column("transfer_gain"))]
        b.fact("corr_knowledge_count_vs_transfer_gain", float(np.corrcoef(kx, tg)[0, 1]) if np.std(kx) > 0 else float("nan"))
    else:
        b.fact("corr_knowledge_count_vs_transfer_gain", float("nan"))
    fg = LC.forgetting_score(curve)
    b.fact("forgetting_score", fg["score"])
    b.fact("regressions", len(an.regressions))
    if an.step:
        b.fact("step_change_at", jsonable(an.step))
    tr = [("transfer gain (the graph that matters)", an.transfer), ("same-year gain", an.same_year), ("stored knowledge count", an.memory),
          ("memorisation gap", an.memorization), ("risk", an.risk)]
    b.table("Trends against experience (per 100 experiences)", ("series", "n", "slope", "lo", "hi", "perm p", "first", "last"),
            [(n, t.n, fmt(t.slope, 5, True), fmt(t.lo, 5, True), fmt(t.hi, 5, True), fmt(t.perm_p, 3), fmt(t.first, 5), fmt(t.last, 5)) for n, t in tr])
    b.table("Curve points", ("step", "experience", "knowledge", "confirmed", "same-year gain", "transfer gain", "risk", "memorisation gap", "as of"),
            [(p.step, p.experience_count, p.knowledge_count, p.validated_knowledge_count, fmt(p.same_year_gain, 5), fmt(p.transfer_gain, 5),
              fmt(p.risk, 5), fmt(p.memorization_gap, 5), p.as_of) for p in curve.points])
    if an.regressions:
        b.table("Regressions (later points below the earlier trend)", ("detail",), [(json.dumps(jsonable(r)),) for r in an.regressions[:10]])
    b.note("The graph read is experience -> future gain (transfer gain). Stored-knowledge growth is used only to recognise hoarding.")
    if an.verdict.value == "INSUFFICIENT_POINTS":
        return b.build(insufficient=True, headline=f"Curve verdict {an.verdict.value}: {defang(an.why)}", claims=claims)
    bad = an.verdict.value in ("DEGRADING", "MEMORISING", "HOARDING", "FLAT")
    return b.build("FAIL" if bad else "OK", f"Curve verdict {an.verdict.value} over {len(curve)} points: {defang(an.why)}", claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K02 transfer (sections 26-27)
# ---------------------------------------------------------------------------------------------------------------
def _bm(d: Any) -> tuple[float, float, float, int]:
    d = d if isinstance(d, Mapping) else {}
    return num(d.get("mean")), num(d.get("lo")), num(d.get("hi")), int(num(d.get("n"), 0))


OK_GATE = ("PASS", "NOT_APPLICABLE")
BAD_TRANSFER = ("OVER_SPECIALISED", "IDENTITY_DEPENDENT", "HARMFUL", "NO_LEARNING", "UNSTABLE")


def transfer_consistency(axes: Mapping[str, Mapping]) -> list[str]:
    """The recorded ratio must equal cross / same. An artefact whose ratio disagrees with its own two means was edited or mis-built."""
    out = []
    for name, a in axes.items():
        sm, _, _, sn = _bm(a.get("same"))
        cm, _, _, cn = _bm(a.get("cross"))
        r = num(a.get("ratio"))
        if math.isfinite(r) and math.isfinite(sm) and math.isfinite(cm) and abs(sm) > 1e-12 and abs(r - cm / sm) > 1e-6 * max(1.0, abs(r)):
            out.append(f"axis {name}: recorded ratio {r:.4f} != cross/same {cm / sm:.4f}")
        if math.isfinite(r) and (sn == 0 or cn == 0):
            out.append(f"axis {name}: a ratio is recorded but a side has no units")
    return out


def report_k02(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K02", "Transfer report", ("K02", "E01"))
    rec, s = ctx.obj("transfer")
    b.source(s)
    if not rec or not isinstance(rec.get("axes"), Mapping) or not rec["axes"]:
        b.absent(s, "transfer report")
        return b.none("no transfer record: unseen-year / regime / stock generalisation is not known")
    axes = rec["axes"]
    b.fact("units", int(num(rec.get("n_units"), 0)))
    b.fact("overall_verdict", str(rec.get("overall", UNMEASURED)))
    tested = [n for n, a in axes.items() if _bm(a.get("cross"))[3] > 0]
    b.fact("axes_recorded", len(axes))
    b.fact("axes_tested", len(tested))
    b.fact("untested_axes", list(rec.get("untested_axes") or []))
    rows = []
    for n, a in sorted(axes.items()):
        sm, slo, shi, sn = _bm(a.get("same"))
        cm, clo, chi, cn = _bm(a.get("cross"))
        mem = _bm(a.get("memorization_gap") and {"mean": a["memorization_gap"].get("gap"), "lo": a["memorization_gap"].get("lo"),
                                                  "hi": a["memorization_gap"].get("hi"), "n": a["memorization_gap"].get("n_a", 0)})
        rows.append((n, a.get("mode", ""), f"{fmt(sm, 5, True)} [{fmt(slo, 4, True)},{fmt(shi, 4, True)}] n={sn}",
                     f"{fmt(cm, 5, True)} [{fmt(clo, 4, True)},{fmt(chi, 4, True)}] n={cn}", fmt(a.get("ratio"), 3), a.get("ratio_status", ""),
                     "yes" if a.get("specialisation") else "no", fmt(mem[0], 5, True), a.get("verdict", ""), a.get("validation", "")))
    b.table("Transfer by axis (gain in familiar vs novel situations, dated after the learner's data)",
            ("axis", "mode", "same context gain", "cross context gain", "ratio", "ratio status", "over-specialised", "memorisation gap", "verdict", "label"), rows)
    if rec.get("depth"):
        b.table("Gain by novelty depth (novel axes at once)", tuple(rec["depth"][0].keys()), [tuple(fmt(v, 5) if isinstance(v, float) else v for v in r.values()) for r in rec["depth"]])
    if rec.get("year_gap"):
        b.table("Gain by years from the nearest trained year", tuple(rec["year_gap"][0].keys()), [tuple(fmt(v, 5) if isinstance(v, float) else v for v in r.values()) for r in rec["year_gap"]])
    inc = transfer_consistency(axes)
    for m in inc:
        b.lost("artefact inconsistent - " + m)
    ident = rec.get("identity")
    if isinstance(ident, Mapping):
        b.fact("identity_gap", num(ident.get("gap")))
        b.fact("identity_gap_ci", [num(ident.get("lo")), num(ident.get("hi"))])
    yr = axes.get("YEAR") or axes.get("year")
    if yr is None:
        b.lost("the unseen-year axis is not in the record: the most important transfer question is unanswered")
    else:
        b.fact("cross_year_gain", _bm(yr.get("cross"))[0])
    verdicts = Counter(str(a.get("verdict", "")) for a in axes.values())
    b.fact("axis_verdicts", dict(verdicts))
    b.note("A ratio far below 1 means the gain lived in the training situations (over-specialised); above 1 needs its interval before it is believed.")
    bad = [n for n, a in axes.items() if str(a.get("verdict", "")) in BAD_TRANSFER] + ([str(rec.get("overall"))] if str(rec.get("overall")) in BAD_TRANSFER else [])
    if inc:
        return b.build("FAIL", f"Transfer artefact contradicts itself ({len(inc)} inconsistencies): not usable", claims=claims)
    if yr is None or not tested or str(rec.get("overall")) == "INSUFFICIENT_EVIDENCE":
        return b.build(insufficient=True, headline=f"Transfer evidence is thin: {len(tested)} axis(es) tested, overall {rec.get('overall')}", claims=claims)
    return b.build("FAIL" if bad else "OK", f"Transfer verdict {rec.get('overall')}; {len(tested)} of {len(axes)} axes tested; failing axes: {', '.join(sorted(set(bad))) or 'none'}", claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K03 memorisation (sections 27-29) and K04 identity gap
# ---------------------------------------------------------------------------------------------------------------
def report_k03(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K03", "Memorization report", ("K03", "H02"))
    card, rec, _ = ctx.latest_card()
    tr, s1 = ctx.obj("transfer")
    crows, s2 = ctx.rows("curve")
    for s in (s1, s2):
        b.source(s)
    gaps: list[tuple[str, float, float, float, int]] = []
    if card is not None:
        m = card.memorization_gap
        if m.measured:
            gaps.append(("scorecard memorisation gap", m.value, m.lo, m.hi, m.n))
        else:
            b.lost("scorecard memorisation gap is UNTESTED")
        C = card.controls.get("C_identity_memoriser")
        b.fact("memoriser_control_run", C is not None)
        b.fact("memoriser_control_flagged", None if C is None else C.detected)
        if C is None or C.detected is not True:
            b.lost("the identity-memoriser control was not run or not flagged: this instrument has not been shown to see a memoriser")
    else:
        b.lost("no scorecard: memorisation gap and the memoriser control are not measured")
    if tr and isinstance(tr.get("axes"), Mapping):
        for n, a in sorted(tr["axes"].items()):
            g = a.get("memorization_gap")
            if isinstance(g, Mapping) and fin(g.get("gap")):
                gaps.append((f"transfer {n}: replay minus forward", num(g["gap"]), num(g.get("lo")), num(g.get("hi")), int(num(g.get("n_a"), 0))))
    mg = [num(r.get("memorization_gap")) for r in crows]
    mg = [v for v in mg if math.isfinite(v)]
    if len(mg) >= 4:
        xs = np.arange(len(mg), dtype=float)
        gaps.append(("curve: latest point", mg[-1], float("nan"), float("nan"), len(mg)))
        b.fact("curve_memorisation_trend_slope_per_point", float(np.polyfit(xs, np.array(mg), 1)[0]))
    elif crows:
        b.lost(f"curve has only {len(mg)} points with a memorisation gap")
    if not gaps:
        return b.none("no memorisation gap is measured anywhere: replay-vs-forward performance is not known")
    b.table("Memorisation gap (performance on already-seen situations minus on later ones; positive = remembering)",
            ("source", "gap", "lo", "hi", "n", "significantly positive"),
            [(n, fmt(g, 5, True), fmt(lo, 5, True), fmt(hi, 5, True), k, "YES" if math.isfinite(lo) and lo > 0 else "no") for n, g, lo, hi, k in gaps])
    flagged = [n for n, g, lo, hi, k in gaps if math.isfinite(lo) and lo > 0]
    b.fact("gaps_measured", len(gaps))
    b.fact("gaps_significantly_positive", len(flagged))
    b.note("A gap near zero is not proof of no memorisation: it counts only if the memoriser control was seen by this instrument (see the numbers).")
    ctrl_blind = card is None or card.controls.get("C_identity_memoriser") is None or card.controls["C_identity_memoriser"].detected is not True
    if flagged:
        return b.build("FAIL", f"Memorisation flagged in {len(flagged)} of {len(gaps)} measured gaps: {'; '.join(flagged[:3])}", claims=claims)
    return b.build("OK", f"No measured memorisation gap is significantly positive ({len(gaps)} measured)" + ("; the instrument is untested against a planted memoriser" if ctrl_blind else ""),
                   insufficient=ctrl_blind and len(gaps) < 2, claims=claims)


def report_k04(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K04", "Identity-gap report", ("K04", "H03"))
    rec, s = ctx.obj("identity")
    b.source(s)
    card, _, _ = ctx.latest_card()
    verdicts = list((rec or {}).get("verdicts") or [])
    if card is not None and card.identity_gap.measured:
        m = card.identity_gap
        b.fact("scorecard_identity_gap", m.value)
        b.fact("scorecard_identity_gap_ci", [m.lo, m.hi])
    if not verdicts:
        b.absent(s, "identity attack verdicts")
        if card is None or not card.identity_gap.measured:
            return b.none("neither identity-attack verdicts nor a scorecard identity gap exist: the learner's dependence on names/dates is not known")
    else:
        b.fact("base_ic", num(rec.get("base_ic")))
        b.fact("deterministic", bool(rec.get("deterministic")))
        b.fact("attacks", len(verdicts))
        collapsed = sorted({f"{v.get('kind')}[{v.get('mode')}]" for v in verdicts if v.get("status") == "COLLAPSE"})
        b.fact("collapsed_under", collapsed)
        b.fact("memorisation_suspected", any(v.get("status") == "COLLAPSE" and v.get("mode") == "eval" for v in verdicts))
        b.fact("logic_identity_dependent", any(v.get("status") == "COLLAPSE" and v.get("mode") == "both" for v in verdicts))
        b.table("Identity attacks (skill kept after names / dates / sequences are disguised)", ("attack", "mode", "status", "retention", "ci low", "ci high", "rows changed"),
                [(v.get("kind"), v.get("mode"), v.get("status"), fmt(v.get("retention"), 3), fmt(v.get("ci_low"), 3), fmt(v.get("ci_high"), 3),
                  fmt(v.get("changed_frac"), 2)) for v in verdicts])
        weak = [v for v in verdicts if num(v.get("changed_frac")) < 0.05]
        if weak:
            b.lost(f"{len(weak)} attack(s) changed under 5% of rows: retention there says nothing (the disguise did not disguise)")
        if not rec.get("deterministic"):
            b.lost("the harness was not deterministic: verdicts cannot be re-derived")
        retention = [num(v.get("retention")) for v in verdicts if fin(v.get("retention"))]
        if retention:
            b.fact("min_retention", min(retention))
            b.fact("median_retention", float(np.median(retention)))
    tr, s2 = ctx.obj("transfer")
    b.source(s2)
    if tr and isinstance(tr.get("identity"), Mapping):
        g = tr["identity"]
        b.table("Transfer report identity gap (familiar identity minus novel identity)", ("gap", "lo", "hi", "flagged"),
                [(fmt(g.get("gap"), 5, True), fmt(g.get("lo"), 5, True), fmt(g.get("hi"), 5, True), g.get("flagged"))])
    dep = bool(verdicts) and (b.facts.get("collapsed_under") or not rec.get("deterministic"))
    ig_bad = card is not None and card.identity_gap.measured and card.identity_gap.lo > 0
    if dep or ig_bad:
        return b.build("FAIL", f"Identity dependence found: collapsed under {b.facts.get('collapsed_under') or 'the scorecard identity gap'}", claims=claims)
    if not verdicts:
        return b.build(insufficient=True, headline="Only the scorecard identity gap exists; no attack was run", claims=claims)
    return b.build("OK", f"Skill survived all {len(verdicts)} identity attacks; nothing here proves the absence of every identity path", claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K05 knowledge health (section 46) - shared with the health dashboard
# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class HealthRow:
    knowledge_id: str
    version: int
    health: Health
    why: str
    epistemic: str
    lifecycle: str
    promotion: str
    truth: float
    reliability: float
    prev_reliability: float
    sample_size: int
    evidence: str
    research: str


def _conf(rec: Mapping, k: str) -> float:
    return num((rec.get("confidence") or {}).get(k))


def classify_health(latest: Mapping, prev: Mapping | None, n_flips: int, contradicted: bool, min_sample: int = 30) -> tuple[Health, str]:
    """One knowledge item's section-46 state and the reason, in a fixed order: the worst true statement wins."""
    ep, lc = str(latest.get("epistemic", "")), str(latest.get("lifecycle", ""))
    truth, rel = _conf(latest, "truth"), _conf(latest, "current_reliability")
    n = int(num((latest.get("evidence") or {}).get("sample_size"), 0))
    if ep == Epistemic.RETIRED.value or lc == Lifecycle.RETIRED.value:
        return Health.DORMANT, "retired (kept, never deleted; not trusted)"
    if ep == Epistemic.CONTRADICTED.value or contradicted:
        return Health.CONTRADICTED, "an unresolved contradiction is recorded against it"
    if lc == Lifecycle.FAILURE.value:
        return Health.BROKEN, "lifecycle FAILURE"
    if n_flips >= 3:
        return Health.UNSTABLE, f"epistemic state changed {n_flips} times across versions"
    if ep == Epistemic.UNKNOWN.value or (not math.isfinite(truth) and not math.isfinite(rel)):
        return Health.UNKNOWN, "no confidence dimension has been measured"
    if lc == Lifecycle.RECOVERY.value:
        return Health.RECOVERING, "lifecycle RECOVERY"
    if lc == Lifecycle.DORMANT.value or ep == Epistemic.GATED.value:
        return Health.DORMANT, "gated / dormant: not active in the current context"
    if ep == Epistemic.DEGRADED.value or lc in (Lifecycle.DEGRADED.value, Lifecycle.DECAY.value):
        return Health.DEGRADING, f"epistemic {ep} / lifecycle {lc}"
    if prev is not None and math.isfinite(rel) and math.isfinite(_conf(prev, "current_reliability")) and rel < _conf(prev, "current_reliability") - 0.1:
        return Health.DEGRADING, f"current reliability fell {_conf(prev, 'current_reliability'):.2f} -> {rel:.2f} between versions"
    if math.isfinite(truth) and math.isfinite(rel) and truth - rel > 0.3:
        return Health.DEGRADING, f"truth {truth:.2f} but current reliability only {rel:.2f}"
    if n < min_sample:
        return Health.INSUFFICIENT_EVIDENCE, f"sample size {n} < {min_sample}"
    return Health.HEALTHY, f"epistemic {ep}, truth {fmt(truth, 2)}, current reliability {fmt(rel, 2)}, n={n}"


def assess_knowledge(ctx: ReportContext) -> tuple[list[HealthRow], list[Source], list[str]]:
    kn, s1 = ctx.rows("knowledge")
    con, s2 = ctx.rows("contradictions")
    qs, s3 = ctx.rows("queue")
    srcs, lost = [s1, s2, s3], []
    if not kn:
        return [], srcs, [f"knowledge: artefact {s1.name} is {s1.status}"]
    chains: dict[str, list[dict]] = defaultdict(list)
    for r in kn:
        chains[str(r.get("knowledge_id"))].append(r)
    open_pairs: dict[tuple, str] = {}
    for c in con:
        open_pairs[tuple(sorted(c.get("pair") or ()))] = str(c.get("verdict", ""))
    contradicted = {k for pair, v in open_pairs.items() if v in ("UNRESOLVED", "UNRESOLVED_KEEP_BOTH", "OPEN", "PARTLY_RESOLVED") for k in pair}
    if not con:
        lost.append(f"contradictions: artefact {s2.name} is {s2.status}: contradiction status is not applied to health")
    rows = []
    for kid, chain in sorted(chains.items()):
        chain.sort(key=lambda r: int(num(r.get("version"), 0)))
        latest, prev = chain[-1], (chain[-2] if len(chain) > 1 else None)
        flips = sum(1 for a, c in zip(chain, chain[1:]) if a.get("epistemic") != c.get("epistemic"))
        h, why = classify_health(latest, prev, flips, kid in contradicted)
        research = ", ".join(str(q.get("id") or q.get("qid") or q.get("question", ""))[:40] for q in qs
                             if kid in json.dumps(q, default=str) and str(q.get("status", "OPEN")).upper() not in ("DONE", "CLOSED", "RESOLVED"))[:120]
        rows.append(HealthRow(kid, int(num(latest.get("version"), 0)), h, why, str(latest.get("epistemic")), str(latest.get("lifecycle")),
                              str(latest.get("promotion")), _conf(latest, "truth"), _conf(latest, "current_reliability"),
                              _conf(prev, "current_reliability") if prev else float("nan"), int(num((latest.get("evidence") or {}).get("sample_size"), 0)),
                              f"v{latest.get('version')} {latest.get('version_reason', '')}", research or "none queued"))
    return rows, srcs, lost


def report_k05(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K05", "Knowledge health report", ("K05", "J11"))
    rows, srcs, lost = assess_knowledge(ctx)
    for s in srcs:
        b.source(s)
    for x in lost:
        b.lost(x)
    if not rows:
        return b.none("no knowledge items: what is trusted, and what is losing trust, is unknown")
    cnt = Counter(r.health.value for r in rows)
    b.fact("items", len(rows))
    for h in Health:
        b.fact(f"health_{h.value.lower()}", cnt.get(h.value, 0))
    b.fact("promotion_states", dict(Counter(r.promotion for r in rows)))
    b.fact("epistemic_states", dict(Counter(r.epistemic for r in rows)))
    b.fact("lifecycle_states", dict(Counter(r.lifecycle for r in rows)))
    trusted = [r for r in rows if r.health == Health.HEALTHY]
    b.fact("trusted_share", share(len(trusted), len(rows)))
    b.fact("trusted_share_wilson", list(wilson_interval(len(trusted), len(rows))))
    b.table("Health of every knowledge item (latest version before now)", ("state", "items", "share"),
            [(h.value, cnt.get(h.value, 0), fmt(share(cnt.get(h.value, 0), len(rows)), 3)) for h in Health])
    b.table("Currently trusted", ("knowledge id", "v", "promotion", "truth", "reliability", "n", "evidence"),
            [(r.knowledge_id, r.version, r.promotion, fmt(r.truth, 2), fmt(r.reliability, 2), r.sample_size, r.why) for r in trusted[:25]])
    losing = [r for r in rows if r.health in (Health.DEGRADING, Health.BROKEN, Health.CONTRADICTED, Health.UNSTABLE)]
    b.table("Losing trust (and why, on what evidence, and what research is on it)", ("knowledge id", "state", "why", "evidence", "research"),
            [(r.knowledge_id, r.health.value, r.why, f"{r.evidence}; n={r.sample_size}", r.research) for r in losing[:40]])
    unstudied = [r for r in losing if r.research == "none queued"]
    b.fact("losing_trust_without_research", len(unstudied))
    if unstudied:
        b.note(f"{len(unstudied)} item(s) are losing trust with nothing queued to investigate them.")
    return b.build(None, f"{len(trusted)} of {len(rows)} knowledge items are HEALTHY; {len(losing)} are losing trust", insufficient=len(rows) < 5, claims=claims)


def health_dashboard(ctx: ReportContext) -> dict:
    """Section-46 dashboard data: what is trusted, what is losing trust, why, on what evidence, and what research is on it."""
    rows, srcs, lost = assess_knowledge(ctx)
    rec = {"section": 46, "now": ctx.now.isoformat(), "code_hash": ctx.code_hash, "sources": [jsonable(s) for s in srcs]}
    if not rows:
        rec.update(status=UNMEASURED, reason="; ".join(lost) or "no knowledge items", trusted=[], losing_trust=[], counts={})
    else:
        pack = lambda r: {"knowledge_id": r.knowledge_id, "version": r.version, "health": r.health.value, "why": r.why, "evidence": r.evidence,
                          "sample_size": r.sample_size, "truth": r.truth, "current_reliability": r.reliability, "research": r.research,
                          "promotion": r.promotion, "epistemic": r.epistemic, "lifecycle": r.lifecycle}
        rec.update(status="PARTIAL" if lost else "MEASURED", reason="; ".join(lost), counts=dict(Counter(r.health.value for r in rows)),
                   trusted=[pack(r) for r in rows if r.health == Health.HEALTHY],
                   losing_trust=[pack(r) for r in rows if r.health in (Health.DEGRADING, Health.BROKEN, Health.CONTRADICTED, Health.UNSTABLE)],
                   other=[pack(r) for r in rows if r.health in (Health.DORMANT, Health.RECOVERING, Health.UNKNOWN, Health.INSUFFICIENT_EVIDENCE)])
    rec["label"] = ValidationLabel.INSUFFICIENT_EVIDENCE.value if rec["status"] == UNMEASURED else ValidationLabel.NOT_VALIDATED.value
    rec["dashboard_id"] = stable_hash(rec)
    return jsonable(rec)


# ---------------------------------------------------------------------------------------------------------------
# K06 failure and K07 recovery (sections 9, 12, 13)
# ---------------------------------------------------------------------------------------------------------------
def report_k06(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K06", "Failure report", ("K06", "D01"))
    rows, s = ctx.rows("failures")
    b.source(s)
    if not rows:
        b.absent(s, "failure classifications")
        return b.none("no classified failures: why the learner loses is unknown")
    n = len(rows)
    causes = Counter(str(r.get("cause", "UNKNOWN")) for r in rows)
    unknown = causes.get(FailureCause.UNKNOWN.value, 0) + causes.get(FailureCause.INSUFFICIENT_EVIDENCE.value, 0)
    named = [r for r in rows if str(r.get("cause")) not in (FailureCause.UNKNOWN.value, FailureCause.INSUFFICIENT_EVIDENCE.value)]
    b.fact("failures_classified", n)
    b.fact("named_cause_share", share(len(named), n))
    b.fact("named_cause_share_wilson", list(wilson_interval(len(named), n)))
    b.fact("unknown_or_insufficient", unknown)
    b.fact("mean_detector_coverage", float(np.nanmean([num(r.get("coverage")) for r in rows])) if any(fin(r.get("coverage")) for r in rows) else float("nan"))
    b.fact("mean_confidence_when_named", float(np.mean([num(r.get("confidence")) for r in named if fin(r.get("confidence"))])) if any(fin(r.get("confidence")) for r in named) else float("nan"))
    b.table("Causes (UNKNOWN is a legitimate answer, not a gap)", ("cause", "failures", "share", "mean confidence"),
            [(c, k, fmt(share(k, n), 3), fmt(np.mean([num(r.get("confidence")) for r in rows if r.get("cause") == c and fin(r.get("confidence"))] or [float("nan")]), 3))
             for c, k in causes.most_common()])
    votes: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for sub, v in (r.get("subsystem_votes") or {}).items():
            votes[sub].append(num(v))
    b.table("Which subsystem each failure teaches (mean net vote)", ("subsystem", "failures with a vote", "mean net vote", "top vote count"),
            [(sub, len(v), fmt(float(np.nanmean(v)), 3, True),
              sum(1 for r in rows if (r.get("subsystem_votes") or {}) and max(r["subsystem_votes"], key=lambda k: num(r["subsystem_votes"][k])) == sub and max(num(x) for x in r["subsystem_votes"].values()) > 0))
             for sub, v in sorted(votes.items())])
    surp = [num(r.get("surprise_bits")) for r in rows if fin(r.get("surprise_bits"))]
    if surp:
        b.fact("mean_surprise_bits", float(np.mean(surp)))
    low_cov = sum(1 for r in rows if fin(r.get("coverage")) and num(r["coverage"]) < 0.5)
    b.fact("classified_with_under_half_coverage", low_cov)
    if low_cov:
        b.note(f"{low_cov} classification(s) ran under half the detectors; their causes are weak.")
    misattrib = [r for r in named if not r.get("meaningful", True)]
    b.fact("named_but_not_meaningful", len(misattrib))
    b.note("The named-cause share is coverage, not accuracy: whether the named causes are RIGHT is measured in K12 (explanation precision).")
    return b.build(None, f"{n} failures classified; {len(named)} named ({fmt(share(len(named), n), 2)}), {unknown} left UNKNOWN/INSUFFICIENT",
                   insufficient=n < 20, claims=claims)


def _states_by_id(rows: Sequence[Mapping]) -> dict[str, str]:
    last: dict[str, tuple[int, str]] = {}
    for r in rows:
        kid, seq = str(r.get("knowledge_id")), int(num(r.get("seq"), 0))
        if kid not in last or seq >= last[kid][0]:
            last[kid] = (seq, str(r.get("to_state")))
    return {k: v[1] for k, v in last.items()}


def transition_id_problems(rows: Sequence[Mapping]) -> list[str]:
    """Each stored transition carries id = stable_hash(body); an edited transition no longer matches."""
    out = []
    for r in rows:
        if not r.get("id"):
            out.append(f"seq {r.get('seq')}: no id")
            continue
        body = {k: v for k, v in r.items() if k != "id"}
        if stable_hash(body, 20) != r["id"]:
            out.append(f"seq {r.get('seq')} ({r.get('knowledge_id')}): content does not match its id (edited)")
    return out


def report_k07(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K07", "Recovery report", ("K07", "J06"))
    rows, s = ctx.rows("retirement")
    b.source(s)
    if not rows:
        b.absent(s, "lifecycle transitions")
        return b.none("no lifecycle transitions recorded: whether retired knowledge can come back is unknown")
    rows = sorted(rows, key=lambda r: (str(r.get("knowledge_id")), int(num(r.get("seq"), 0))))
    problems = transition_id_problems(rows)
    for p in problems[:5]:
        b.lost("integrity - " + p)
    states = _states_by_id(rows)
    b.fact("transitions", len(rows))
    b.fact("items_with_transitions", len(states))
    b.fact("current_states", dict(Counter(states.values())))
    kinds = Counter(str(r.get("kind")) for r in rows)
    b.table("Transitions by kind", ("kind", "count"), kinds.most_common())
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_item[str(r.get("knowledge_id"))].append(r)
    recov, relapse, days_down = [], 0, []
    for kid, tl in by_item.items():
        down_at = None
        for i, t in enumerate(tl):
            if str(t.get("to_state")) in ("DEGRADED", "DORMANT", "RETIRED") and down_at is None:
                down_at = t
            elif str(t.get("to_state")) == "ACTIVE" and down_at is not None:
                recov.append((kid, down_at, t))
                try:
                    days_down.append((as_date(t["at"]) - as_date(down_at["at"])).days)
                except (KeyError, ValueError):
                    pass
                if any(str(x.get("to_state")) in ("DEGRADED", "RETIRED") for x in tl[i + 1:]):
                    relapse += 1
                down_at = None
    b.fact("recoveries", len(recov))
    b.fact("recoveries_that_relapsed", relapse)
    b.fact("relapse_share", share(relapse, len(recov)))
    if days_down:
        b.fact("median_days_down", float(np.median(days_down)))
    down_causes = Counter(str(r.get("cause", "UNKNOWN")) for r in rows if str(r.get("to_state")) in ("DEGRADED", "DORMANT", "RETIRED"))
    b.table("Why knowledge went down", ("cause", "transitions"), down_causes.most_common())
    b.table("Recoveries (down -> ACTIVE)", ("knowledge id", "went down", "from", "recovered", "reason"),
            [(k, d.get("at"), d.get("to_state"), t.get("at"), t.get("reason", "")) for k, d, t in recov[:25]])
    retired_now = [k for k, v in states.items() if v == "RETIRED"]
    b.fact("retired_now", len(retired_now))
    b.note("RETIRED is not deleted: retired items stay in the store and can recover only through a recorded recovery condition.")
    if problems:
        return b.build("FAIL", f"Lifecycle ledger fails its integrity check ({len(problems)} edited transitions): recoveries cannot be trusted", claims=claims)
    return b.build(None, f"{len(recov)} recoveries across {len(states)} items; {relapse} relapsed", insufficient=len(rows) < 10, claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K08 contradictions, K09 missed winners, K10 experiment memory
# ---------------------------------------------------------------------------------------------------------------
def report_k08(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K08", "Contradiction report", ("K08", "J09"))
    rows, s = ctx.rows("contradictions")
    b.source(s)
    if not rows:
        b.absent(s, "contradiction ledger")
        return b.none("no contradiction investigations recorded: disagreement between knowledge items is not being tracked")
    latest: dict[tuple, dict] = {}
    tries: Counter = Counter()
    for r in sorted(rows, key=lambda r: str(r.get("at"))):
        pair = tuple(sorted(r.get("pair") or ()))
        latest[pair] = r
        tries[pair] += 1
    verdicts = Counter(str(r.get("verdict")) for r in latest.values())
    open_states = ("UNRESOLVED", "UNRESOLVED_KEEP_BOTH", "OPEN", "PARTLY_RESOLVED", "NEEDS_DATA")
    b.fact("investigations", len(rows))
    b.fact("pairs", len(latest))
    b.fact("latest_verdicts", dict(verdicts))
    b.fact("open_pairs", sum(v for k, v in verdicts.items() if k in open_states))
    b.fact("resolved_by_context", verdicts.get("RESOLVED_BY_CONTEXT", 0) + verdicts.get("RESOLVED", 0))
    b.fact("spurious_split", verdicts.get("SPURIOUS_SPLIT", 0))
    m = [int(num(r.get("m_tests"), 0)) for r in rows]
    b.fact("cumulative_tests_max", max(m) if m else 0)
    b.fact("pairs_investigated_more_than_once", sum(1 for v in tries.values() if v > 1))
    b.table("Pairs and their newest verdict", ("pair", "verdict", "dimension", "adjusted p", "tests so far", "investigations", "data through"),
            [(" vs ".join(p), r.get("verdict"), r.get("dimension", ""), fmt(r.get("p_adj"), 4), r.get("m_tests"), tries[p], r.get("rows_through", ""))
             for p, r in sorted(latest.items(), key=lambda kv: (str(kv[1].get("verdict")) not in open_states, kv[0]))[:40]])
    weak = [p for p, r in latest.items() if fin(r.get("p_adj")) and 0.05 < num(r["p_adj"]) <= 1 and str(r.get("verdict")) == "RESOLVED_BY_CONTEXT"]
    if weak:
        b.note(f"{len(weak)} pair(s) are marked resolved-by-context with an adjusted p above 0.05: resolution is not established.")
    b.note("Both sides of an open contradiction are kept; neither is deleted because the other was more recent.")
    return b.build(None, f"{len(latest)} contradictory pairs investigated; {b.facts['open_pairs']} still open", insufficient=len(latest) < 3, claims=claims)


def report_k09(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K09", "Missed-winner report", ("K09", "D07"))
    rec, s = ctx.obj("missed")
    b.source(s)
    if not rec:
        b.absent(s, "missed-winner analysis")
        return b.none("no missed-winner analysis: what the filters throw away is unknown")
    reasons, dists, wf = rec.get("reasons") or [], rec.get("distinctions") or [], rec.get("walk_forward") or {}
    b.fact("reasons_analysed", len(reasons))
    b.fact("distinctions_tested", len(dists))
    costly = [r for r in reasons if r.get("verdict") == "COSTLY"]
    b.fact("costly_filters", [r.get("reason") for r in costly])
    if reasons:
        b.table("Rejection reasons: win rate of REJECTED candidates vs all rejected (every reject counted, not only the missed winners)",
                ("reason", "rejected", "win rate", "ci low", "ci high", "base rate", "z", "q", "verdict"),
                [(r.get("reason"), r.get("n"), fmt(r.get("win_rate"), 3), fmt((r.get("ci") or [None, None])[0], 3), fmt((r.get("ci") or [None, None])[1], 3),
                  fmt(r.get("base_rate"), 3), fmt(r.get("z"), 2, True), fmt(r.get("q"), 4), r.get("verdict")) for r in reasons])
    else:
        b.lost("no rejection-reason table in the artefact")
    if dists:
        passed = [d for d in dists if str(d.get("verdict")) == "PASS"]
        b.fact("distinctions_passed_out_of_sample", len(passed))
        b.table("Distinctions between missed winners and picks (out-of-sample verdict)", ("rule", "lift", "mean gain", "p", "excess vs random", "weeks", "verdict"),
                [(d.get("description") or d.get("did"), fmt(d.get("lift"), 2), fmt(d.get("mean_gain"), 5, True), fmt(d.get("p_value"), 4),
                  fmt(d.get("excess_vs_random"), 5, True), d.get("n_weeks"), d.get("verdict")) for d in dists])
    else:
        b.lost("no distinctions were tested out of sample")
    if wf:
        disc, passed = int(num(wf.get("discovered"), 0)), int(num(wf.get("passed"), 0))
        b.fact("walk_forward_discovered", disc)
        b.fact("walk_forward_passed", passed)
        b.fact("walk_forward_pass_share", share(passed, disc))
        b.fact("walk_forward_pass_share_wilson", list(wilson_interval(passed, disc)))
    null = rec.get("null_control")
    if isinstance(null, Mapping):
        b.fact("null_control", jsonable(null))
    else:
        b.lost("no shuffled-label null control: a distinction that also 'passes' on shuffled outcomes would go unseen")
    b.note("A missed winner is only a lesson if the rule that would have caught it also loses less on the names it would have wrongly added.")
    if not reasons and not dists:
        return b.none("the missed-winner artefact holds neither reasons nor distinctions")
    return b.build(None, f"{len(reasons)} rejection reasons ({len(costly)} costly), {len(dists)} distinctions tested out of sample",
                   insufficient=not dists or not isinstance(null, Mapping), claims=claims)


def _norm_question(q: str) -> str:
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", str(q).lower()))))


def report_k10(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K10", "Experiment-memory report", ("K10", "G01"))
    rows, s = ctx.rows("experiments")
    b.source(s)
    if not rows:
        b.absent(s, "experiment ledger")
        return b.none("no experiment records: what has already been tried is unknown, so it will be repeated")
    n = len(rows)
    status = Counter(str(r.get("status")) for r in rows)
    kinds = Counter(str((r.get("result") or {}).get("kind", "NO_RESULT")) for r in rows)
    b.fact("experiments", n)
    b.fact("status", dict(status))
    b.fact("result_kinds", dict(kinds))
    b.fact("legacy_records", sum(1 for r in rows if r.get("legacy")))
    answered = [r for r in rows if r.get("status") == "ANSWERED"]
    no_lesson = [r.get("experiment_id") for r in answered if not r.get("not_learned")]
    null_no_power = [r.get("experiment_id") for r in rows if (r.get("result") or {}).get("kind") == "NULL" and not (r.get("result") or {}).get("power_note")]
    b.fact("answered_without_what_was_not_learned", len(no_lesson))
    b.fact("null_results_without_power", len(null_no_power))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_norm_question(r.get("question", ""))].append(r)
    repeats = {k: v for k, v in groups.items() if len(v) > 1}
    unjustified = [r.get("experiment_id") for v in repeats.values() for r in v[1:] if not r.get("repeat_reason")]
    b.fact("questions_asked_more_than_once", len(repeats))
    b.fact("repeats_without_a_reason", len(unjustified))
    bits = [num((r.get("belief_update") or {}).get("surprise_bits")) for r in rows if r.get("belief_update")]
    moved = [num((r.get("belief_update") or {}).get("moved")) for r in rows if r.get("belief_update")]
    bits, moved = [v for v in bits if math.isfinite(v)], [v for v in moved if math.isfinite(v)]
    if bits:
        b.fact("mean_surprise_bits", float(np.mean(bits)))
        b.fact("mean_belief_movement", float(np.mean(moved)) if moved else float("nan"))
        b.fact("experiments_that_moved_no_belief", sum(1 for v in moved if v < 0.02))
    pairs = [(num((r.get("prediction") or {}).get("confidence")), 1.0 if (r.get("result") or {}).get("kind") == "CONFIRMED" else 0.0)
             for r in rows if r.get("result") and (r.get("result") or {}).get("kind") in ("CONFIRMED", "REFUTED") and fin((r.get("prediction") or {}).get("confidence"))]
    if len(pairs) >= 10:
        p, y = np.array([a for a, _ in pairs]), np.array([c for _, c in pairs])
        b.fact("prediction_brier", float(np.mean((p - y) ** 2)))
        b.fact("prediction_base_rate_brier", float(np.mean((y.mean() - y) ** 2)))
        b.fact("predictions_scored", len(pairs))
    else:
        b.lost(f"only {len(pairs)} experiments have both a stated confidence and a decisive result: the system's own predictions cannot be calibrated yet")
    b.table("Status", ("status", "count"), status.most_common())
    b.table("Questions asked more than once", ("question (normalised)", "times", "reason for repeating"),
            [(k[:80], len(v), "; ".join(sorted({str(r.get('repeat_reason') or 'NONE') for r in v[1:]}))) for k, v in list(repeats.items())[:20]])
    tags = Counter(t for r in rows for t in (r.get("tags") or []))
    b.table("Most used tags", ("tag", "experiments"), tags.most_common(10))
    if no_lesson or null_no_power or unjustified:
        b.note("Records with a missing 'what was not learned', a null result with no stated power, or an unexplained repeat are defects in the memory itself.")
    return b.build("FAIL" if (null_no_power or no_lesson) else None, f"{n} experiments; {len(repeats)} questions repeated ({len(unjustified)} without a reason)",
                   insufficient=n < 10, claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K11 research priority + J15 dashboard (section 51)
# ---------------------------------------------------------------------------------------------------------------
PRIORITY_WEIGHTS = {"uncertainty": 0.20, "decision_impact": 0.20, "transfer_potential": 0.15, "failure_evidence": 0.15,
                    "contradiction": 0.10, "surprise": 0.10}
ALIASES = {"decision_impact": ("decision_impact", "decision_value", "stake"), "failure_evidence": ("failure_evidence", "failure"),
           "uncertainty": ("uncertainty",), "transfer_potential": ("transfer_potential", "transfer"), "contradiction": ("contradiction",),
           "surprise": ("surprise", "unexplained_surprise"), "feasibility": ("feasibility",)}


def _factor(item: Mapping, name: str) -> float:
    src = item.get("factors") if isinstance(item.get("factors"), Mapping) else item
    for a in ALIASES[name]:
        if fin(src.get(a)):
            return min(1.0, max(0.0, num(src[a])))
    if name == "feasibility" and fin(src.get("cost")):
        return 1.0 / (1.0 + max(0.0, num(src["cost"])))
    return float("nan")


def priority_of(item: Mapping) -> tuple[float, list[str], str]:
    """Section-51 priority for one queue item: weighted benefit of the factors that are PRESENT (weights renormalised over them, missing
    ones named), times feasibility. A recorded score wins; the source is always stated."""
    missing = [k for k in PRIORITY_WEIGHTS if not math.isfinite(_factor(item, k))]
    if fin(item.get("score")):
        return num(item["score"]), missing, "recorded"
    have = {k: w for k, w in PRIORITY_WEIGHTS.items() if k not in missing}
    if not have:
        return float("nan"), missing, "no factors"
    val = sum(_factor(item, k) * w for k, w in have.items()) / sum(have.values())
    feas = _factor(item, "feasibility")
    return val * (feas if math.isfinite(feas) else 1.0), missing + ([] if math.isfinite(feas) else ["feasibility"]), "recomputed"


def research_priority_dashboard(ctx: ReportContext, top: int = 15) -> dict:
    """J15 dashboard data: the ranked research queue with the factor behind every rank, staleness, and how the queue has evolved."""
    items, s = ctx.rows("queue")
    rec = {"section": 51, "checklist": "J15", "now": ctx.now.isoformat(), "code_hash": ctx.code_hash, "sources": [jsonable(s)]}
    if not items:
        rec.update(status=UNMEASURED, reason=f"artefact queue is {s.status}" + (f" ({s.error})" if s.error else ""), ranked=[], targets={}, evolution=[])
    else:
        ranked = []
        for it in items:
            p, missing, how = priority_of(it)
            ranked.append({"id": str(it.get("id") or it.get("qid") or stable_hash(it, 8)), "question": defang(it.get("question") or it.get("text", "")),
                           "target": str(it.get("target", "")), "status": str(it.get("status", "OPEN")).upper(), "priority": p, "score_source": how,
                           "missing_factors": missing, "enqueued_at": str(it.get("enqueued_at", "")), "attempts": int(num(it.get("attempts"), 0)),
                           "factors": {k: _factor(it, k) for k in list(PRIORITY_WEIGHTS) + ["feasibility"]}})
        openq = sorted([r for r in ranked if r["status"] in ("OPEN", "QUEUED", "RUNNING", "PENDING") and math.isfinite(r["priority"])],
                       key=lambda r: (-r["priority"], r["id"]))
        by_month: Counter = Counter(r["enqueued_at"][:7] for r in ranked if r["enqueued_at"])
        last = max((as_date(r["enqueued_at"]) for r in ranked if r["enqueued_at"]), default=None)
        rec.update(status="MEASURED", reason="", ranked=openq[:top], n_items=len(ranked), n_open=len(openq),
                   statuses=dict(Counter(r["status"] for r in ranked)), targets=dict(Counter(r["target"] for r in openq)),
                   evolution=sorted(by_month.items()), days_since_newest_item=(ctx.now - last).days if last else None,
                   stale=bool(last and (ctx.now - last).days > 60), items_with_missing_factors=sum(1 for r in ranked if r["missing_factors"]),
                   repeatedly_attempted=[r["id"] for r in ranked if r["attempts"] >= 3])
    rec["label"] = ValidationLabel.INSUFFICIENT_EVIDENCE.value if rec["status"] == UNMEASURED else ValidationLabel.NOT_VALIDATED.value
    rec["dashboard_id"] = stable_hash(rec)
    return jsonable(rec)


def report_k11(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K11", "Research-priority report", ("K11", "J15"))
    dash = research_priority_dashboard(ctx)
    b.source(Source(**{k: dash["sources"][0].get(k, v) for k, v in dataclasses.asdict(Source("", "", "")).items()}))
    if dash["status"] == UNMEASURED:
        b.lost(dash["reason"])
        return b.none("no research queue: what to investigate next is not being decided from evidence")
    for k in ("n_items", "n_open", "statuses", "targets", "days_since_newest_item", "stale", "items_with_missing_factors", "repeatedly_attempted"):
        b.fact(k, dash[k])
    b.table("Top of the queue (priority = weighted uncertainty, decision impact, transfer, failure, contradiction, surprise; x feasibility)",
            ("rank", "id", "target", "priority", "source", "uncertainty", "impact", "transfer", "failure", "contradiction", "surprise", "feasibility", "missing"),
            [(i + 1, r["id"], r["target"], fmt(r["priority"], 3), r["score_source"], *(fmt(r["factors"][k], 2) for k in list(PRIORITY_WEIGHTS) + ["feasibility"]),
              ",".join(r["missing_factors"]) or "-") for i, r in enumerate(dash["ranked"])])
    b.table("Queue growth by month enqueued", ("month", "items"), dash["evolution"])
    if dash["stale"]:
        b.note(f"No item was enqueued in the last {dash['days_since_newest_item']} days: a queue that does not evolve from experience is not learning.")
    if dash["items_with_missing_factors"]:
        b.note(f"{dash['items_with_missing_factors']} item(s) are ranked with factors missing; missing factors are excluded, never counted as zero.")
    return b.build("FAIL" if dash["stale"] else None, f"{dash['n_open']} open research questions of {dash['n_items']}; queue {'STALE' if dash['stale'] else 'active'}",
                   insufficient=dash["n_items"] < 3, claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K12 meta-learning (section 37)
# ---------------------------------------------------------------------------------------------------------------
def report_k12(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    from . import meta_learning as ML
    b = Builder(ctx, "K12", "Meta-learning report", ("K12", "G12"))
    rec, s = ctx.obj("meta")
    b.source(s)
    if not rec:
        b.absent(s, "meta-learning records")
        return b.none("no meta-learning records: the system has not measured how well its own learning process works")
    late = 0
    for k in ML.MetaStore.KINDS:                                   # a label resolved at/after now was not known at now
        keep = [d for d in rec.get(k) or [] if not d.get("resolved_at") or as_date(d["resolved_at"]) < ctx.now]
        late += len(rec.get(k) or []) - len(keep)
        rec[k] = keep
    if late and ctx.strict_future:
        raise FirewallBreach(f"meta_learning: {late} record(s) resolved at/after now={ctx.now}")
    b.fact("records_dropped_resolved_after_now", late)
    counts = {k: len(rec.get(k) or []) for k in ML.MetaStore.KINDS}
    b.fact("record_counts", counts)
    disc = rec.get("discoveries") or []
    resolved = [d for d in disc if d.get("survived_oos") is not None]
    b.fact("discoveries_resolved", len(resolved))
    if resolved:
        surv = sum(1 for d in resolved if d["survived_oos"])
        b.fact("out_of_sample_survival_rate", share(surv, len(resolved)))
        b.fact("out_of_sample_survival_wilson", list(wilson_interval(surv, len(resolved))))
        fam = defaultdict(lambda: [0, 0])
        for d in resolved:
            fam[str(d.get("family"))][1] += 1
            fam[str(d.get("family"))][0] += int(bool(d["survived_oos"]))
        b.table("Which pattern families survive out of sample", ("family", "resolved", "survived", "rate", "wilson low", "wilson high"),
                [(f, n, k, fmt(share(k, n), 3), fmt(wilson_interval(k, n)[0], 3), fmt(wilson_interval(k, n)[1], 3)) for f, (k, n) in sorted(fam.items())])
        ratios = [(num(d.get("oos_effect")) / num(d.get("is_effect"))) for d in resolved if fin(d.get("is_effect")) and fin(d.get("oos_effect")) and abs(num(d["is_effect"])) > 1e-12]
        if ratios:
            b.fact("median_oos_to_is_effect_ratio", float(np.median(ratios)))
    else:
        b.lost("no discovery has a resolved out-of-sample label")
    lr = rec.get("learners") or []
    if lr:
        claimed = [x for x in lr if num(x.get("claimed_gain")) > 0]
        fake = [x for x in claimed if not x.get("verified")]
        b.fact("learners_claiming_gain", len(claimed))
        b.fact("claimed_gain_not_verified", len(fake))
        b.fact("fake_improvement_rate", share(len(fake), len(claimed)))
        b.table("Learners: claimed vs independently verified", ("learner", "claimed gain", "verified", "verified gain"),
                [(x.get("learner"), fmt(x.get("claimed_gain"), 5, True), x.get("verified"), fmt(x.get("verified_gain"), 5, True)) for x in lr[:25]])
    else:
        b.lost("no learner claimed-vs-verified records: fake gains cannot be measured")
    ex = rec.get("explanations") or []
    if ex:
        named = [e for e in ex if e.get("predicted_cause") not in ("UNKNOWN", "INSUFFICIENT_EVIDENCE")]
        right = [e for e in named if e.get("predicted_cause") == e.get("verified_cause") and e.get("verified_cause") != "UNKNOWN"]
        b.fact("explanations_named", len(named))
        b.fact("explanation_precision", share(len(right), len(named)))
        b.fact("explanation_precision_wilson", list(wilson_interval(len(right), len(named))))
        per = defaultdict(lambda: [0, 0])
        for e in named:
            per[e["predicted_cause"]][1] += 1
            per[e["predicted_cause"]][0] += int(e["predicted_cause"] == e.get("verified_cause"))
        b.table("Which failure explanations tend to be right", ("predicted cause", "asserted", "verified right", "precision"),
                [(c, n, k, fmt(share(k, n), 3)) for c, (k, n) in sorted(per.items())])
    else:
        b.lost("no explanation records: whether the failure explanations are correct is unknown")
    dec = rec.get("decays") or []
    if dec:
        by = defaultdict(list)
        for d in dec:
            by[str(d.get("memory_type"))].append(d)
        rows = []
        for mt, ds in sorted(by.items()):
            age = np.array([num(d.get("age_days")) for d in ds])
            ratio = np.array([num(d.get("reliability_ratio")) for d in ds])
            ok = np.isfinite(age) & np.isfinite(ratio) & (ratio > 0)
            rate = float(np.polyfit(age[ok], np.log(ratio[ok]), 1)[0]) if ok.sum() >= 3 and np.ptp(age[ok]) > 0 else float("nan")
            rows.append((mt, len(ds), fmt(rate * 30, 4, True), fmt(math.log(2) / -rate, 0) if math.isfinite(rate) and rate < 0 else "n/a"))
        b.table("Which memory types decay (log-reliability slope per 30 days; half-life in days)", ("memory type", "records", "slope/30d", "half-life"), rows)
    ab = rec.get("ablations") or []
    if ab:
        fails = Counter(a.get("feature_family") for a in ab if a.get("failed"))
        b.table("Feature families that failed ablation", ("family", "failures"), fails.most_common(10))
    yl = rec.get("yields") or []
    if yl:
        by = defaultdict(list)
        for y in yl:
            by[str(y.get("kind"))].append(y)
        b.table("Which experiments produce useful knowledge", ("kind", "runs", "useful share", "bits per compute-minute"),
                [(k, len(v), fmt(share(sum(1 for y in v if y.get("useful")), len(v)), 3),
                  fmt(sum(num(y.get("gain_bits"), 0.0) for y in v) / max(1e-9, sum(num(y.get("cost_minutes"), 0.0) for y in v)), 4)) for k, v in sorted(by.items())])
    b.note("Meta-learning is itself evaluated out of sample or not at all: only records whose label was known before now are read.")
    n_all = sum(counts.values())
    return b.build(None, f"{n_all} meta-learning records; {len(resolved)} discoveries have an out-of-sample label", insufficient=len(resolved) < 30 and not ex and not lr,
                   claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K13 promotion / rejection (section 45)
# ---------------------------------------------------------------------------------------------------------------
def promotion_inconsistencies(rows: Sequence[Mapping]) -> list[str]:
    """A decision that says PROMOTE while a critical gate failed (or a gate list is empty) is not a decision but a bug or an edit."""
    out = []
    for r in rows:
        gates = r.get("gates") or []
        crit_fail = [g.get("gate") for g in gates if g.get("critical") and str(g.get("status")) not in OK_GATE]
        if str(r.get("verdict")) == "PROMOTE" and crit_fail:
            out.append(f"{r.get('knowledge_id')} v{r.get('version')}: PROMOTE despite failed critical gate(s) {crit_fail}")
        if str(r.get("verdict")) == "PROMOTE" and not gates:
            out.append(f"{r.get('knowledge_id')} v{r.get('version')}: PROMOTE with no gate results")
        if str(r.get("verdict")) == "PROMOTE" and r.get("preconditions"):
            out.append(f"{r.get('knowledge_id')} v{r.get('version')}: PROMOTE with failed preconditions {r.get('preconditions')}")
    return out


def report_k13(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K13", "Promotion/rejection report", ("K13", "J04"))
    rows, s = ctx.rows("promotions")
    b.source(s)
    if not rows:
        b.absent(s, "promotion decisions")
        return b.none("no promotion decisions: what reached production and what was refused is unknown")
    n = len(rows)
    verdict = Counter(str(r.get("verdict")) for r in rows)
    b.fact("decisions", n)
    b.fact("promoted", verdict.get("PROMOTE", 0))
    b.fact("blocked", verdict.get("BLOCK", 0))
    b.fact("promotion_share", share(verdict.get("PROMOTE", 0), n))
    gate_fail: Counter = Counter()
    gate_crit: Counter = Counter()
    gate_seen: Counter = Counter()
    for r in rows:
        for g in r.get("gates") or []:
            gate_seen[g.get("gate")] += 1
            if str(g.get("status")) not in OK_GATE:
                gate_fail[g.get("gate")] += 1
                if g.get("critical"):
                    gate_crit[g.get("gate")] += 1
    b.table("Gates: how often each one stops a candidate", ("gate", "evaluated", "failed", "fail share", "critical failures"),
            [(g, gate_seen[g], gate_fail[g], fmt(share(gate_fail[g], gate_seen[g]), 3), gate_crit[g]) for g in sorted(gate_seen, key=lambda g: -gate_fail[g])])
    blocked = [r for r in rows if r.get("verdict") == "BLOCK"]
    b.table("Recent rejections and why", ("knowledge id", "v", "as of", "failed gates", "preconditions"),
            [(r.get("knowledge_id"), r.get("version"), r.get("now"), ",".join(g.get("gate", "") for g in r.get("gates", []) if str(g.get("status")) not in OK_GATE),
              ",".join(map(str, r.get("preconditions") or []))) for r in sorted(blocked, key=lambda r: str(r.get("now")))[-20:]])
    repeat = Counter(str(r.get("knowledge_id")) for r in blocked)
    b.fact("blocked_three_or_more_times", sum(1 for v in repeat.values() if v >= 3))
    k, ks = ctx.rows("knowledge")
    b.source(ks)
    if k:
        latest = {}
        for r in sorted(k, key=lambda r: int(num(r.get("version"), 0))):
            latest[r.get("knowledge_id")] = r
        b.fact("knowledge_by_promotion_state", dict(Counter(str(r.get("promotion")) for r in latest.values())))
        promoted_ids = {r.get("knowledge_id") for r in rows if r.get("verdict") == "PROMOTE"}
        skipped = [i for i, r in latest.items() if str(r.get("promotion")) == "CHAMPION" and i not in promoted_ids]
        b.fact("champions_without_a_recorded_promotion", len(skipped))
        if skipped:
            b.note(f"{len(skipped)} knowledge item(s) are CHAMPION with no recorded PROMOTE decision: the gate was bypassed or its record is missing.")
    else:
        b.lost("knowledge store missing: champions cannot be cross-checked against decisions")
    bad = promotion_inconsistencies(rows)
    for m in bad[:6]:
        b.lost("inconsistent decision - " + m)
    b.fact("inconsistent_decisions", len(bad))
    if bad or b.facts.get("champions_without_a_recorded_promotion"):
        return b.build("FAIL", f"Promotion record is not trustworthy: {len(bad)} inconsistent decisions, {b.facts.get('champions_without_a_recorded_promotion', 0)} unrecorded champions", claims=claims)
    return b.build(None, f"{verdict.get('PROMOTE', 0)} promoted, {verdict.get('BLOCK', 0)} blocked of {n} decisions", insufficient=n < 5, claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K14 provenance audit (sections 5, 28-30, 49, 56)
# ---------------------------------------------------------------------------------------------------------------
def provenance_findings(rec: Mapping, now: dt.date) -> list[str]:
    pv = rec.get("provenance")
    if not isinstance(pv, Mapping):
        return ["no provenance block"]
    names = {f.name for f in dataclasses.fields(Provenance)}
    try:
        p = Provenance(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pv.items() if k in names})
    except TypeError as e:
        return [f"provenance malformed: {e}"]
    out = p.check()
    try:
        if not p.could_exist_at(now):
            out.append(f"learned_at {p.learned_at} / outcomes seen through {p.outcomes_seen_through or p.learned_at} is not before now={now}: future memory")
    except ValueError as e:
        out.append(f"provenance dates unreadable: {e}")
    if not p.data_hash:
        out.append("data_hash missing: the data this came from cannot be identified")
    return out


def report_k14(ctx: ReportContext, claims: Sequence[str] = ()) -> Report:
    b = Builder(ctx, "K14", "Provenance audit report", ("K14", "H05"))
    kn, s = ctx.rows("knowledge")
    b.source(s)
    cards, s2, problems = ctx.cards()
    b.source(s2)
    if not kn and not cards:
        b.absent(s, "knowledge store")
        b.absent(s2, "scorecard store")
        return b.none("nothing to audit: neither knowledge items nor scorecards exist")
    findings: list[tuple[str, str]] = []
    if kn:
        chains: dict[str, list[dict]] = defaultdict(list)
        for r in kn:
            chains[str(r.get("knowledge_id"))].append(r)
        latest = {}
        for kid, ch in chains.items():
            ch.sort(key=lambda r: int(num(r.get("version"), 0)))
            vs = [int(num(r.get("version"), 0)) for r in ch]
            if vs != list(range(1, len(ch) + 1)):
                findings.append((kid, f"version chain is not 1..{len(ch)}: {vs}"))
            for a, c in zip(ch, ch[1:]):
                if str(c.get("updated_at", "")) < str(a.get("updated_at", "")):
                    findings.append((kid, f"v{c.get('version')} dated before its parent"))
            if ch and str(ch[0].get("parent_hash", "")):
                findings.append((kid, "version 1 carries a parent_hash"))
            latest[kid] = ch[-1]
            for m in provenance_findings(ch[-1], ctx.now):
                findings.append((kid, m))
        hashes = {str((r.get("provenance") or {}).get("code_hash", "")) for r in latest.values()}
        b.fact("knowledge_items", len(latest))
        b.fact("distinct_code_hashes", len(hashes - {""}))
        b.fact("items_with_clean_provenance", len(latest) - len({k for k, _ in findings}))
        b.fact("items_with_findings", len({k for k, _ in findings}))
        ex_ids = [i for i, r in latest.items() if any(x in json.dumps(r.get("contexts", {})) for x in ("ticker", "date", "symbol"))]
        b.fact("items_whose_contexts_name_a_ticker_or_date", len(ex_ids))
        for i in ex_ids:
            findings.append((i, "context conditions reference an identity key (ticker/date/symbol): identity leak risk"))
    else:
        b.lost("no knowledge items: provenance of learned items is unaudited")
    b.fact("scorecard_store_problems", len(problems))
    for p in problems:
        findings.append(("scorecards", p))
    if cards:
        cc = Counter(str(c.get("code_hash")) for c in cards)
        b.fact("scorecard_versions", len(cards))
        b.fact("scorecard_code_hashes", dict(cc))
        b.fact("scorecard_versions_without_seed", sum(1 for c in cards if c.get("seed") is None))
        ch = {str(c.get("controls_hash")) for c in cards if c.get("controls_hash")}
        b.fact("control_epochs", len(ch))
        if len(ch) > 1:
            findings.append(("scorecards", f"the control learners changed {len(ch) - 1} time(s): gains before and after are not comparable"))
        for c in cards:
            if c.get("seed") is None or not c.get("code_hash"):
                findings.append((str(c.get("learner_version")), "scorecard without seed or code hash"))
    else:
        b.lost("no scorecards: their provenance is unaudited")
    if ctx.checklist_path and ctx.checklist_path.exists():
        try:
            items = json.loads(ctx.checklist_path.read_text(encoding="utf-8")).get("items", [])
        except (OSError, ValueError):
            items = []
            b.lost("checklist unreadable")
        overclaim = [i.get("id") for i in items if i.get("status") == "VALIDATED" and not any((ROOT / e).exists() or Path(e).exists() for e in i.get("evidence", []) if isinstance(e, str))]
        missing_code = [i.get("id") for i in items if i.get("status") in ("IMPLEMENTED", "VALIDATED")
                        for cp in i.get("code_paths", []) if not (ROOT / str(cp).split("::")[0].split(" (")[0]).exists()]
        b.fact("checklist_items", len(items))
        b.fact("checklist_validated_without_evidence", overclaim)
        b.fact("checklist_implemented_with_missing_code_path", sorted(set(missing_code))[:20])
        for i in overclaim:
            findings.append((str(i), "checklist says VALIDATED but no evidence file exists"))
    else:
        b.lost("master checklist not found: claimed statuses were not cross-checked")
    b.table("Findings", ("subject", "finding"), findings[:60])
    b.fact("findings", len(findings))
    b.note("A provenance audit that finds nothing is credible only if it can find something: see the planted-defect tests of this module.")
    if findings:
        return b.build("FAIL", f"{len(findings)} provenance findings across {len({k for k, _ in findings})} subjects", claims=claims)
    return b.build("OK", "no provenance defect found in the stores that exist", insufficient=not kn, claims=claims)


# ---------------------------------------------------------------------------------------------------------------
# K15 full learning-system report and the section-65 delta
# ---------------------------------------------------------------------------------------------------------------
def delta_table(ctx: ReportContext) -> tuple[list[tuple], Source]:
    rec, s = ctx.obj("delta")
    rows = []
    for d in DELTA_DIMS:
        v = (rec or {}).get(d) if rec else None
        if isinstance(v, Mapping) and fin(v.get("value")):
            lo, hi = num(v.get("lo")), num(v.get("hi"))
            rows.append((d, fmt(v["value"], 5, True), fmt(lo, 5, True), fmt(hi, 5, True), "positive" if lo > 0 else "negative" if hi < 0 else "not separated from zero"))
        else:
            rows.append((d, UNMEASURED, "", "", "not measured"))
    return rows, s


def worst_label(labels: Sequence[ValidationLabel]) -> ValidationLabel:
    if not labels:
        return ValidationLabel.INSUFFICIENT_EVIDENCE
    return max(labels, key=lambda l: LABEL_ORDER[l])


def report_k15(ctx: ReportContext, claims: Sequence[str] = (), parts: Mapping[str, Report] | None = None) -> Report:
    b = Builder(ctx, "K15", "Full learning-system report", ("K15", "L01"))
    parts = dict(parts) if parts is not None else {k: g(ctx) for k, g in GENERATORS.items() if k != "K15"}
    decision, card, problems = ctx.decision()
    rows = [(k, r.title, r.status, r.label.value, defang(r.headline)[:110]) for k, r in sorted(parts.items())]
    b.table("Every component report", ("key", "report", "status", "label", "headline"), rows)
    unmeasured = [k for k, r in parts.items() if r.status == UNMEASURED]
    failed = [k for k, r in parts.items() if r.label == ValidationLabel.FAILED_VALIDATION]
    b.fact("reports", len(parts))
    b.fact("unmeasured_reports", unmeasured)
    b.fact("failed_reports", failed)
    b.fact("weakest_label", worst_label([r.label for r in parts.values()]).value)
    if card is not None:
        b.table("Scorecard (section 47)", ("field", "value"), [(f, _cell_of(getattr(card, f))) for f in SC.SECTION_47_FIELDS + ("learning_gain",)])
        cm = SC.claim_matrix(card)
        b.table("Which fields may be described as better (needs the field's own controls)", tuple(cm.columns),
                [tuple(str(x) for x in r) for r in cm.itertuples(index=False)])
        b.fact("claimable_fields", SC.claimable_fields(card))
        b.fact("still_needed_for_a_claim", SC.missing_for_claim(card))
        b.fact("gate_blockers", list(decision.blockers) if decision else [])
        b.fact("untested_fields", card.untested_fields())
    else:
        b.lost("no scorecard: section 47 is UNMEASURED, so no field can be described as better")
    drows, ds = delta_table(ctx)
    b.source(ds)
    b.table("Learning delta by dimension (section 65; one dimension never proves the system)", ("dimension", "delta", "lo", "hi", "reading"), drows)
    for k in unmeasured:
        b.lost(f"{k} {parts[k].title} is UNMEASURED")
    cl, cs = ctx.checklist_path, None
    if cl and cl.exists():
        try:
            items = json.loads(cl.read_text(encoding="utf-8"))["items"]
            cs = Counter(i.get("status") for i in items)
            b.fact("checklist_status", dict(cs))
        except (OSError, ValueError, KeyError):
            b.lost("checklist unreadable")
    else:
        b.lost("master checklist not found")
    b.note("This report never upgrades an implemented component to a higher label: that needs a sealed window and blind reruns, which this generator does not run.")
    b.note("Headline of the whole system is the weakest component label above, not the average.")
    measured = bool(parts) and len(unmeasured) < len(parts)
    return b.build("FAIL" if failed else None, f"{len(parts) - len(unmeasured)} of {len(parts)} component reports measured; weakest label {b.facts['weakest_label']}",
                   measured=measured, insufficient=bool(unmeasured) and not failed, claims=claims)


def _cell_of(v: Any) -> str:
    if isinstance(v, SC.Measured):
        return f"{v.value:+.5f} [{v.lo:+.4f}, {v.hi:+.4f}] n={v.n}" if v.measured and math.isfinite(v.lo) else f"{v.value:+.5f}" if v.measured else v.status.value
    if isinstance(v, SC.ComputeCost):
        return f"cpu {v.cpu_seconds:.1f}s wall {v.wall_seconds:.1f}s" if v.measured else "UNTESTED"
    return getattr(v, "value", str(v))


GENERATORS: dict[str, Callable[..., Report]] = {
    "K01": report_k01, "K02": report_k02, "K03": report_k03, "K04": report_k04, "K05": report_k05, "K06": report_k06, "K07": report_k07,
    "K08": report_k08, "K09": report_k09, "K10": report_k10, "K11": report_k11, "K12": report_k12, "K13": report_k13, "K14": report_k14,
    "K15": report_k15,
}
FILENAMES = {"K01": "learning_curve", "K02": "transfer", "K03": "memorization", "K04": "identity_gap", "K05": "knowledge_health", "K06": "failure",
             "K07": "recovery", "K08": "contradiction", "K09": "missed_winner", "K10": "experiment_memory", "K11": "research_priority",
             "K12": "meta_learning", "K13": "promotion_rejection", "K14": "provenance_audit", "K15": "full_learning_system"}


def generate_all(ctx: ReportContext, claims: Sequence[str] = ()) -> dict[str, Report]:
    out = {k: g(ctx, claims) for k, g in GENERATORS.items() if k != "K15"}
    out["K15"] = report_k15(ctx, claims, parts=out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# writing and auditing the output directory
# ---------------------------------------------------------------------------------------------------------------
def _write(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()[:16]


def write_all(ctx: ReportContext, out_dir: Path | str = DEFAULT_OUT, claims: Sequence[str] = ()) -> dict:
    """Generate everything and write markdown + JSON per report, the two dashboard data files, and index.json (file hashes)."""
    out = Path(out_dir)
    reps = generate_all(ctx, claims)
    files: dict[str, str] = {}
    for k, r in reps.items():
        stem = f"{k}_{FILENAMES[k]}"
        files[f"{stem}.md"] = _write(out / f"{stem}.md", r.to_markdown() + "\n")
        files[f"{stem}.json"] = _write(out / f"{stem}.json", r.to_json() + "\n")
    for name, data in (("J15_research_priority_dashboard.json", research_priority_dashboard(ctx)), ("S46_health_dashboard.json", health_dashboard(ctx))):
        files[name] = _write(out / name, json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    index = {"now": ctx.now.isoformat(), "code_hash": ctx.code_hash, "seed": ctx.seed,
             "reports": {k: {"label": r.label.value, "status": r.status, "report_id": r.to_record()["report_id"]} for k, r in reps.items()}, "files": files}
    _write(out / "index.json", json.dumps(index, indent=1, sort_keys=True) + "\n")
    return index


def audit_output(out_dir: Path | str = DEFAULT_OUT, ctx: ReportContext | None = None) -> list[str]:
    """Fail-closed audit of a written directory: every report present, hashes match, UNMEASURED reports carry an INSUFFICIENT EVIDENCE
    label and their reason, no report is VALIDATED without checklist evidence, no forbidden word survives in the markdown."""
    out, problems = Path(out_dir), []
    idx = out / "index.json"
    if not idx.exists():
        return ["index.json missing: nothing to audit"]
    try:
        index = json.loads(idx.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"index.json unreadable: {e}"]
    for k in GENERATORS:
        if k not in index.get("reports", {}):
            problems.append(f"{k} missing from the index")
    for name, h in index.get("files", {}).items():
        p = out / name
        if not p.exists():
            problems.append(f"{name} listed but absent")
        elif sha_of(p) != h:
            problems.append(f"{name} was modified after it was written")
    for k, meta in index.get("reports", {}).items():
        jp = out / f"{k}_{FILENAMES.get(k, '')}.json"
        if not jp.exists():
            continue
        rec = json.loads(jp.read_text(encoding="utf-8"))
        if rec.get("status") == UNMEASURED and (rec.get("label") != ValidationLabel.INSUFFICIENT_EVIDENCE.value or not str(rec.get("headline", "")).startswith(UNMEASURED)):
            problems.append(f"{k}: UNMEASURED but not labelled/announced as such")
        if rec.get("label") == ValidationLabel.VALIDATED.value and ctx is not None and not checklist_validated(ctx, k)[0]:
            problems.append(f"{k}: labelled VALIDATED without checklist evidence")
        if rec.get("label") not in {l.value for l in ValidationLabel}:
            problems.append(f"{k}: unknown label {rec.get('label')!r}")
        md = out / f"{k}_{FILENAMES.get(k, '')}.md"
        if md.exists():
            text = md.read_text(encoding="utf-8")
            body = re.sub(r"## Claim status.*?(?=\n## )", "", text, count=1, flags=re.S)
            hits = SC.claim_violations(body.replace(ValidationLabel.VALIDATED.value, ""), SC.ClaimDecision(False, (), ValidationLabel.INSUFFICIENT_EVIDENCE))
            if hits:
                problems.append(f"{k}: markdown asserts {hits}")
    return problems
