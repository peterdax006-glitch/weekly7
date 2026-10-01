"""Creator K10 - evaluation and improvement measurement (C77 secs 30-32, 35, 49-56; package CR06) - IMPLEMENTED, NOT VALIDATED.

The one place a development change is MEASURED. An "arm" is a solver plus the configuration that defines it; `compare()` runs a
baseline arm and a candidate arm on the sealed devbench under identical conditions and turns the scores into ledger records:

    dev split, R replicates per arm   -> Measurement(solve_rate) x R per arm                (benefit + reproducibility)
    the same runs                     -> Measurement(no_false_completion), (no_regression)  (guard metrics = regression check)
    holdout split, once per arm,      -> Measurement(solve_rate, split=HOLDOUT)             (holdout; refused unless the
    only for frozen configurations                                                            candidate config was frozen first)
    all of the above                  -> ImprovementClaim, verdict COMPUTED by creator.model:improvement_verdict

Every Measurement cites the saved per-task score file as evidence (path + sha256) and a ComputationRef naming this module's code
hash and the digests of its inputs and output, so the ledger can re-check the numbers' origin and recompute the verdict. A paired
per-task table (solved by both / only base / only candidate / neither) is kept alongside, because a solve-rate difference on a
dozen tasks is weak evidence and the claim must say so (C77 sec 71 uncertainty)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from creator import devbench as D
from creator import model as M
from creator.ledger import REPO_ROOT, Ledger

PRIMARY = "solve_rate"
GUARDS = ("no_false_completion", "no_regression")
EVAL_DIR = REPO_ROOT / "state" / "creator" / "evals"
FROZEN = REPO_ROOT / "state" / "creator" / "frozen_configs.json"


class EvaluationError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ provenance of numbers

def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def module_code_hash(*modules: Any) -> str:
    h = hashlib.sha256()
    for mod in modules:
        h.update(Path(mod.__file__).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def computation(function: str, inputs: Any, output: Any, record_ids: Sequence[str] = ()) -> M.ComputationRef:
    import creator.evaluate as E
    return M.ComputationRef(function=function, code_hash=module_code_hash(E, D, M), inputs_sha256=_sha(inputs),
                            output_sha256=_sha(output), record_ids=tuple(record_ids))


# ------------------------------------------------------------------------------------------------ arms and metrics

SolverFn = Callable[[Mapping[str, Any], Path], D.SolverResult]


@dataclasses.dataclass(frozen=True)
class Arm:
    """A solver and the configuration that defines it (the configuration is what gets frozen before the holdout)."""
    name: str
    solver: SolverFn
    config: Mapping[str, Any]

    @property
    def config_digest(self) -> str:
        return hashlib.sha256(json.dumps(dict(self.config), sort_keys=True).encode()).hexdigest()[:16]


def rate_with_se(k: int, n: int) -> tuple[float, float]:
    """Proportion and its standard error with a small-sample floor: 0/n or n/n still carries uncertainty (Agresti-Coull style
    +2/+4 adjustment for the SE only), so a perfect score on few tasks is never treated as certain."""
    if n <= 0:
        raise EvaluationError("no tasks scored")
    p = k / n
    pa = (k + 2) / (n + 4)
    return p, math.sqrt(pa * (1 - pa) / (n + 4))


def metrics_of(score: D.SuiteScore) -> dict[str, tuple[float, float, int]]:
    n = len(score.scores)
    solved = sum(s.outcome == "SOLVED" for s in score.scores)
    false_c = sum(s.outcome == "FALSE_COMPLETION" for s in score.scores)
    regress = sum(s.outcome == "REGRESSION" for s in score.scores)
    return {PRIMARY: (*rate_with_se(solved, n), n),
            "no_false_completion": (*rate_with_se(n - false_c, n), n),
            "no_regression": (*rate_with_se(n - regress, n), n)}


def pool(scores: Sequence[D.SuiteScore]) -> D.SuiteScore:
    allt = tuple(s for sc in scores for s in sc.scores)
    return D.SuiteScore(scores[0].split, scores[0].solver, allt)


@dataclasses.dataclass(frozen=True)
class Paired:
    both: int
    only_base: int
    only_candidate: int
    neither: int

    @property
    def sign_test_p(self) -> float:
        """Two-sided exact sign test on the discordant tasks (McNemar exact)."""
        n = self.only_base + self.only_candidate
        if n == 0:
            return 1.0
        k = min(self.only_base, self.only_candidate)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return min(1.0, 2 * tail)


def paired(base: D.SuiteScore, cand: D.SuiteScore) -> Paired:
    def solved(sc: D.SuiteScore) -> dict[str, float]:
        tot: dict[str, list[int]] = {}
        for s in sc.scores:
            tot.setdefault(s.task_id, []).append(1 if s.outcome == "SOLVED" else 0)
        return {k: sum(v) / len(v) for k, v in tot.items()}
    b, c = solved(base), solved(cand)
    keys = sorted(set(b) & set(c))
    both = sum(1 for k in keys if b[k] >= 0.5 and c[k] >= 0.5)
    ob = sum(1 for k in keys if b[k] >= 0.5 > c[k])
    oc = sum(1 for k in keys if c[k] >= 0.5 > b[k])
    return Paired(both, ob, oc, len(keys) - both - ob - oc)


# ------------------------------------------------------------------------------------------------ the comparison

@dataclasses.dataclass(frozen=True)
class EvaluationReport:
    experiment_id: str
    claim_id: str
    verdict: M.Verdict
    detail: Mapping[str, Any]
    paired_dev: Paired
    paired_holdout: Optional[Paired]
    base_dev: Mapping[str, Any]
    cand_dev: Mapping[str, Any]
    evidence_dir: str
    seconds: float


def _save(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return path


def _measure(ledger: Ledger, experiment_id: str, metric: str, value: float, se: float, n: int, population: str,
             conditions: str, split: M.Split, evidence: Path, inputs: Any) -> str:
    rec = M.Measurement(created_by=M.Role.VALIDATOR, parents=(experiment_id,), metric=metric, value=value, stderr=se, n=n,
                        population=population, conditions=conditions, higher_is_better=True, split=split,
                        computation=computation("creator.evaluate:metrics_of", inputs, [value, se, n]),
                        evidence=(M.EvidenceRef.of(evidence, ledger.evidence_root, "devbench_scores"),))
    return ledger.append(rec)


def compare(ledger: Ledger, experiment_id: str, baseline: Arm, candidate: Arm, replicates: int = 2,
            tasks: Optional[Sequence[D.Task]] = None, manifest: Optional[Mapping[str, Any]] = None,
            frozen_path: Path = FROZEN, out_dir: Optional[Path] = None, use_holdout: bool = True,
            min_effect: float = 0.0, z: float = 2.0, scratch: Optional[Path] = None) -> EvaluationReport:
    """Measure candidate against baseline under identical conditions and append Measurements + the ImprovementClaim."""
    if replicates < 2:
        raise EvaluationError("at least two replicates per arm: one run cannot show reproducibility")
    if ledger.get(experiment_id).RTYPE != "Experiment":
        raise EvaluationError(f"{experiment_id} is not an Experiment")
    t0 = time.monotonic()
    m = manifest or D.load_manifest()
    ts = list(tasks or D.load_tasks())
    out = (out_dir or EVAL_DIR) / experiment_id
    conditions = f"devbench manifest {str(m['digest'])[:16]}; scorer {module_code_hash(D)[:16]}; tasks {len(ts)}"
    pop = {"dev": f"devbench:{str(m['digest'])[:16]}:dev", "holdout": f"devbench:{str(m['digest'])[:16]}:holdout"}
    if use_holdout:
        D.freeze_config({"arm": baseline.name, **dict(baseline.config)}, frozen_path)
        D.freeze_config({"arm": candidate.name, **dict(candidate.config)}, frozen_path)

    runs: dict[str, list[D.SuiteScore]] = {"base": [], "cand": []}
    base_ids: list[str] = []
    cand_ids: list[str] = []
    for r in range(replicates):                                         # interleaved, so drift hits both arms alike
        for key, arm, ids in (("base", baseline, base_ids), ("cand", candidate, cand_ids)):
            sc = D.score_dev(arm.solver, arm.name, tasks=ts, manifest=m, scratch=scratch)
            runs[key].append(sc)
            ev = _save(out / f"{key}_dev_r{r}.json", [s.to_dict() for s in sc.scores])
            v, se, n = metrics_of(sc)[PRIMARY]
            ids.append(_measure(ledger, experiment_id, PRIMARY, v, se, n, pop["dev"], conditions, M.Split.DEV, ev,
                                {"arm": arm.name, "config": arm.config_digest, "replicate": r}))
    pb, pc = pool(runs["base"]), pool(runs["cand"])
    guard_b: list[str] = []
    guard_c: list[str] = []
    for g in GUARDS:
        for key, sc, ids in (("base", pb, guard_b), ("cand", pc, guard_c)):
            ev = _save(out / f"{key}_dev_pooled.json", [s.to_dict() for s in sc.scores])
            v, se, n = metrics_of(sc)[g]
            ids.append(_measure(ledger, experiment_id, g, v, se, n, pop["dev"], conditions, M.Split.DEV, ev,
                                {"arm": key, "guard": g}))
    hold_b = hold_c = None
    ph: Optional[Paired] = None
    if use_holdout:
        hs: dict[str, D.SuiteScore] = {}
        for key, arm in (("base", baseline), ("cand", candidate)):
            sc = D.score_holdout(arm.solver, {"arm": arm.name, **dict(arm.config)}, frozen_path, arm.name, tasks=ts, manifest=m,
                                 scratch=scratch)
            hs[key] = sc
            ev = _save(out / f"{key}_holdout.json", [s.to_dict() for s in sc.scores])
            v, se, n = metrics_of(sc)[PRIMARY]
            mid = _measure(ledger, experiment_id, PRIMARY, v, se, n, pop["holdout"], conditions, M.Split.HOLDOUT, ev,
                           {"arm": arm.name, "holdout": True})
            if key == "base":
                hold_b = mid
            else:
                hold_c = mid
        ph = paired(hs["base"], hs["cand"])
    get = ledger.get
    verdict, detail = M.improvement_verdict([get(i) for i in base_ids], [get(i) for i in cand_ids],   # type: ignore[misc]
                                            [(get(b), get(c)) for b, c in zip(guard_b, guard_c)],   # type: ignore[misc]
                                            (get(hold_b), get(hold_c)) if hold_b and hold_c else None,  # type: ignore[arg-type]
                                            min_effect=min_effect, z=z)
    pd = paired(pb, pc)
    detail = dict(detail, paired_dev=dataclasses.asdict(pd), sign_test_p=pd.sign_test_p)
    all_ids = base_ids + cand_ids + guard_b + guard_c + [i for i in (hold_b, hold_c) if i]
    claim = M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(experiment_id,), subject_id=experiment_id,
                               baseline_ids=tuple(base_ids), candidate_ids=tuple(cand_ids), verdict=verdict,
                               computation=computation(M.VERDICT_FUNCTION, all_ids, verdict.value, all_ids),
                               regression_baseline_ids=tuple(guard_b), regression_candidate_ids=tuple(guard_c),
                               holdout_baseline_id=hold_b, holdout_candidate_id=hold_c, min_effect=min_effect, z=z)
    cid = ledger.append(claim)
    _save(out / "report.json", {"claim": cid, "verdict": verdict.value, "detail": detail,
                                "base": baseline.name, "candidate": candidate.name})
    return EvaluationReport(experiment_id, cid, verdict, detail, pd, ph, pb.summary(), pc.summary(), str(out),
                            round(time.monotonic() - t0, 1))
