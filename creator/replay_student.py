"""Replay: re-apply a teacher solution that was never measured (Nupen's curriculum; 2 Oct 2026).

When a package was solved by the teacher but its cycle ended without a verdict - pulled back for RAM, stopped by the owner's
pause, an error - the solution is still in the lesson log (files_before -> files_after). When the same objective is planned again
and the files it changed are still byte-identical to what the teacher started from, this student re-applies the solution, so the
kernel can finally measure it instead of the teacher solving the same package twice. It judges nothing: the kernel's measurement
decides, and its adoptions count as the TEACHER's work (curriculum.TEACHER), never as a student's skill.

A solution the kernel already measured - adopted or rejected - is never replayed."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from creator import curriculum as C

NAME = C.REPLAY


def _replayable(les: C.Lesson) -> bool:
    if les.solver != C.CLAUDE or not les.claimed_done or not les.files_after:
        return False
    if les.adopted is None:
        return True                                                    # the cycle never reported a verdict
    return not C.is_skill_signal(les)                                   # cancelled / error: never measured


class ReplayStudent:
    name = NAME

    def __init__(self, lessons_path: Path) -> None:
        self.log = C.LessonLog(Path(lessons_path))

    def _candidates(self, objective: str) -> list[C.Lesson]:
        if not objective:
            return []
        measured = {les.objective for les in self.log.lessons() if les.adopted is not None and C.is_skill_signal(les)
                    and les.solver in C.TEACHER and les.objective == objective}
        if measured:
            return []                                                   # the kernel already judged this objective's solution
        return [les for les in reversed(self.log.lessons()) if les.objective == objective and _replayable(les)]

    def can_attempt(self, task: Any) -> bool:
        return bool(self._candidates(str(getattr(task, "objective", ""))))

    @staticmethod
    def _matches(workdir: Path, before: dict[str, str]) -> bool:
        for rel, text in before.items():
            p = workdir / rel
            if text == "":
                if p.exists():
                    return False
            elif not p.is_file() or p.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n") != text.replace("\r\n", "\n"):
                return False
        return True

    def __call__(self, plan: Any, package: Any, workdir: Path) -> Any:
        from creator.kernel import WorkResult
        workdir = Path(workdir)
        les: Optional[C.Lesson] = next((x for x in self._candidates(str(getattr(package, "objective", "")))
                                        if self._matches(workdir, x.files_before)), None)
        if les is None:
            return WorkResult(False, "no unmeasured teacher solution matches the current files")
        root = workdir.resolve()
        for rel in les.files_after:                                    # a lesson log is data: never write outside the sandbox
            if Path(rel).is_absolute() or not (root / rel).resolve().is_relative_to(root):
                return WorkResult(False, f"lesson {les.lesson_id} names a path outside the work directory: {rel!r}")
        for rel, text in les.files_after.items():
            p = workdir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode("utf-8"))
        return WorkResult(True, f"replayed teacher lesson {les.lesson_id} ({les.package_id}): {', '.join(sorted(les.files_after))}",
                          by=NAME, reasoning=les.reasoning)
