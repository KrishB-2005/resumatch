"""Measure the matcher against hand-labelled cases.

"I improved the matcher" is not a claim anyone can check. This turns it into a
number that moves when the behaviour moves, over cases where a careful human
reader would agree on the answer.

Two rules keep it honest:

Failures stay in the file. A benchmark you prune until it passes measures the
pruning, not the matcher. `data/benchmark/cases.json` keeps every case that was
ever labelled, and the reported score is over all of them.

Labels say *why*. Each case carries the reasoning a human would give, so a
failure can be argued with — sometimes the right fix is the label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from resumatch import skills
from resumatch.matching import match_requirement
from resumatch.models import Evidence, Requirement, RequirementMatch, Resume

CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "cases.json"


@dataclass(frozen=True)
class Case:
    id: str
    requirement: str
    resume: list[str]
    matched: bool
    why: str
    method: str | None = None
    scoreable: bool = True


@dataclass(frozen=True)
class Result:
    case: Case
    match: RequirementMatch

    @property
    def matched_ok(self) -> bool:
        return self.match.matched is self.case.matched

    @property
    def scoreable_ok(self) -> bool:
        return self.match.scoreable is self.case.scoreable

    @property
    def method_ok(self) -> bool:
        """Only checked when the case bothers to state it."""
        if self.case.method is None:
            return True
        return self.match.method.value == self.case.method

    @property
    def passed(self) -> bool:
        return self.matched_ok and self.scoreable_ok and self.method_ok

    @property
    def failure(self) -> str:
        if not self.scoreable_ok:
            return (
                f"scoreable={self.match.scoreable}, expected {self.case.scoreable}"
            )
        if not self.matched_ok:
            return f"matched={self.match.matched}, expected {self.case.matched}"
        return f"method={self.match.method.value}, expected {self.case.method}"


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Case(**entry) for entry in raw["cases"]]


def run_case(case: Case) -> Result:
    requirement = Requirement(
        text=case.requirement, skills=skills.extract_skills(case.requirement)
    )
    resume = Resume(
        evidence=[
            Evidence(text=line, skills=skills.extract_skills(line))
            for line in case.resume
        ]
    )
    return Result(case=case, match=match_requirement(requirement, resume))


def run(path: Path = CASES_PATH) -> list[Result]:
    return [run_case(case) for case in load_cases(path)]


def format_results(results: list[Result], *, verbose: bool = False) -> str:
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    lines = ["", f"  {len(passed)}/{len(results)} cases", ""]

    for result in results if verbose else failed:
        mark = "ok  " if result.passed else "FAIL"
        lines.append(f"  {mark} {result.case.id}")
        if not result.passed:
            lines.append(f"         {result.failure}")
            lines.append(f"         expected because: {result.case.why}")

    if not failed:
        lines.append("  no failures")
    lines.append("")
    return "\n".join(lines)
