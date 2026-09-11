"""Measure the parser against whole documents.

The matching benchmark starts from clean sentences that somebody already
extracted. That is not where a score actually goes wrong. It goes wrong when a
two-column PDF hides the "Requirements" heading and a sixteen-requirement
posting parses as three, or when a wrapped bullet becomes two requirements and
the second one is a fragment naming no skill. Everything downstream then works
perfectly on the wrong input and reports a confident number.

So these cases are whole postings and resumes in the shapes people actually
send them, labelled with what a careful reader says should come out. As with
the matching suite, failing cases stay in the file — a benchmark pruned until
it passes measures the pruning.

Assertions are deliberately loose about wording and strict about structure:
`contains` rather than exact text, because how a requirement is phrased is the
document's business, while how many there are and which bucket they land in is
the parser's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from resumatch.models import Importance, JobSpec, Resume
from resumatch.parsing import parse_job, parse_resume, read_text

CASES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "benchmark" / "parsing" / "cases.json"
)


@dataclass(frozen=True)
class ParseCase:
    id: str
    kind: str
    file: str
    why: str
    expect: dict[str, Any]


@dataclass
class ParseResult:
    case: ParseCase
    failures: list[str] = field(default_factory=list)
    parsed: JobSpec | Resume | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


def load_cases(path: Path = CASES_PATH) -> list[ParseCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ParseCase(**entry) for entry in raw["cases"]]


def _check_counts(
    expect: dict[str, Any], counts: dict[str, int], out: list[str]
) -> None:
    for key, actual in counts.items():
        wanted = expect.get(key)
        if wanted is not None and actual != wanted:
            out.append(f"{key}: got {actual}, expected {wanted}")


def _check_absent(expect: dict[str, Any], texts: list[str], out: list[str]) -> None:
    """Company prose and contact details must not become scoreable items."""
    for phrase in expect.get("must_not_include", []):
        hit = next((t for t in texts if phrase.lower() in t.lower()), None)
        if hit is not None:
            out.append(f'must_not_include {phrase!r} but parsed: "{hit[:52]}"')


def _check_skills_anywhere(
    expect: dict[str, Any], found: set[str], out: list[str]
) -> None:
    missing = [s for s in expect.get("skills_anywhere", []) if s not in found]
    if missing:
        out.append(f"skills never found anywhere: {', '.join(missing)}")


def _run_job(case: ParseCase, text: str) -> ParseResult:
    job = parse_job(text)
    out: list[str] = []
    expect = case.expect

    if (want := expect.get("title_contains")) and want.lower() not in job.title.lower():
        out.append(f"title: got {job.title[:46]!r}, expected to contain {want!r}")
    if (want := expect.get("company_contains")) and want.lower() not in (
        job.company.lower()
    ):
        out.append(f"company: got {job.company!r}, expected to contain {want!r}")

    required = [r for r in job.requirements if r.importance is Importance.REQUIRED]
    preferred = [r for r in job.requirements if r.importance is Importance.PREFERRED]
    _check_counts(
        expect,
        {
            "requirement_count": len(job.requirements),
            "required_count": len(required),
            "preferred_count": len(preferred),
        },
        out,
    )

    for wanted in expect.get("must_include", []):
        phrase = wanted["contains"]
        match = next(
            (r for r in job.requirements if phrase.lower() in r.text.lower()), None
        )
        if match is None:
            out.append(f"no requirement containing {phrase!r}")
            continue
        if (imp := wanted.get("importance")) and match.importance.value != imp:
            out.append(f"{phrase!r}: importance {match.importance.value}, expected {imp}")
        if (kind := wanted.get("kind")) and match.kind.value != kind:
            out.append(f"{phrase!r}: kind {match.kind.value}, expected {kind}")
        if (years := wanted.get("years")) is not None and match.years != years:
            out.append(f"{phrase!r}: years {match.years}, expected {years}")
        for skill in wanted.get("skills", []):
            if skill not in match.skills:
                out.append(
                    f"{phrase!r}: skill {skill!r} missing (got {match.skills or 'none'})"
                )

    _check_absent(expect, [r.text for r in job.requirements], out)
    _check_skills_anywhere(
        expect, {s for r in job.requirements for s in r.skills}, out
    )
    return ParseResult(case=case, failures=out, parsed=job)


def _run_resume(case: ParseCase, text: str) -> ParseResult:
    resume = parse_resume(text)
    out: list[str] = []
    expect = case.expect

    if (minimum := expect.get("evidence_min")) and len(resume.evidence) < minimum:
        out.append(f"evidence lines: got {len(resume.evidence)}, expected >= {minimum}")

    present = {item.section.value for item in resume.evidence}
    for section in expect.get("sections_present", []):
        if section not in present:
            out.append(f"section {section!r} never assigned to any line")

    for wanted in expect.get("lines", []):
        phrase = wanted["contains"]
        match = next(
            (e for e in resume.evidence if phrase.lower() in e.text.lower()), None
        )
        if match is None:
            out.append(f"no evidence line containing {phrase!r}")
            continue
        if (section := wanted.get("section")) and match.section.value != section:
            out.append(
                f"{phrase!r}: section {match.section.value}, expected {section}"
            )
        for skill in wanted.get("skills", []):
            if skill not in match.skills:
                out.append(
                    f"{phrase!r}: skill {skill!r} missing (got {match.skills or 'none'})"
                )

    _check_absent(expect, [e.text for e in resume.evidence], out)
    _check_skills_anywhere(expect, resume.skills(), out)
    return ParseResult(case=case, failures=out, parsed=resume)


def run_case(case: ParseCase, root: Path | None = None) -> ParseResult:
    base = root or CASES_PATH.parent
    text = read_text(base / case.file)
    if case.kind == "job":
        return _run_job(case, text)
    return _run_resume(case, text)


def run(path: Path = CASES_PATH) -> list[ParseResult]:
    return [run_case(case, path.parent) for case in load_cases(path)]


def format_results(results: list[ParseResult], *, verbose: bool = False) -> str:
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    broken = sum(len(r.failures) for r in failed)
    lines = [
        "",
        f"  {len(passed)}/{len(results)} documents"
        + (f"  ({broken} failed checks)" if failed else ""),
        "",
    ]

    for result in results if verbose else failed:
        mark = "ok  " if result.passed else "FAIL"
        lines.append(f"  {mark} {result.case.id}")
        if not result.passed:
            for failure in result.failures:
                lines.append(f"         {failure}")
            lines.append(f"         labelled because: {result.case.why}")
            lines.append("")

    if not failed:
        lines.append("  no failures")
    lines.append("")
    return "\n".join(lines)
