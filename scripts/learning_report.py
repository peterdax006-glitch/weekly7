"""Write the learning reports K01-K15, the J15 research-priority dashboard data and the section-46 health dashboard data
(contract C62 sections 46-48, 59, 65, 79) to state/research/learning_reports/, then audit what was written (fail closed).

  python scripts/learning_report.py --now 2030-01-01 [--root state/research/learning] [--out state/research/learning_reports]
         [--seed 0] [--claim "learning improved"] [--strict-future] [--audit-only]

Everything is computed from the artefacts under --root and the master checklist; a missing artefact makes its report say
UNMEASURED. --claim asks the generator to attach a sentence and prints whether the claim gate accepted or refused it.
Exit code 0 = written and audit clean; 2 = the audit found problems (the files are still written so the problem can be read)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.learning import reports as R  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--now", required=True, help="explicit as-of date; nothing dated on/after it is read")
    ap.add_argument("--root", default=str(R.DEFAULT_INPUTS))
    ap.add_argument("--out", default=str(R.DEFAULT_OUT))
    ap.add_argument("--checklist", default=str(R.DEFAULT_CHECKLIST))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--claim", action="append", default=[])
    ap.add_argument("--strict-future", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ctx = R.ReportContext(a.root, a.now, seed=a.seed, checklist=a.checklist, strict_future=a.strict_future)
    if not a.audit_only:
        index = R.write_all(ctx, a.out, claims=a.claim)
        for k, m in sorted(index["reports"].items()):
            print(f"{k}  {m['status']:<10} {m['label']}")
        if a.claim:
            for c in R.generate_all(ctx, a.claim)["K15"].claims:
                print(f"claim {'ACCEPTED' if c.accepted else 'REFUSED'}: {c.text!r} ({c.reason})")
    problems = R.audit_output(a.out, ctx)
    for p in problems:
        print("AUDIT:", p)
    print(f"audit: {'CLEAN' if not problems else str(len(problems)) + ' problem(s)'}  -> {a.out}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
