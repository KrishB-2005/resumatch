"""Build the match report, including deterministic advice.

The advice here is rule-based, not generated. Each rule fires off a fact the
matcher already established, so every suggestion names the requirement it
closes and can be checked against the resume. That is the difference between
"consider highlighting your cloud experience" and "the posting says k8s twice;
your resume only says Kubernetes — a keyword filter will miss it".

The semantic pass in `agents.py` adds to this; it does not replace it.
"""

from __future__ import annotations

from resumatch import matching
from resumatch.llm.client import LLMClient
from resumatch.models import (
    Importance,
    JobSpec,
    MatchMethod,
    MatchReport,
    RequirementMatch,
    Resume,
    Suggestion,
)

# Below this, the resume is being read for a different job than it was written
# for, and line-level edits are the wrong advice.
POOR_FIT_THRESHOLD = 0.5

# A resume carrying many skills the posting never mentions is not wrong, but on
# a thirty-second read those lines compete with the ones that matter.
NOISE_THRESHOLD = 6


def _alias_suggestions(matches: list[RequirementMatch]) -> list[Suggestion]:
    """Matches that only survived because of the alias table.

    These are the highest-value finding in the whole report. The human reading
    the resume will make the connection; an automated keyword filter sitting in
    front of them will not.
    """
    out: list[Suggestion] = []
    for match in matches:
        # Named per skill, not per requirement. "Docker and Kubernetes" can be
        # literal on Docker and a synonym on Kubernetes at the same time, and
        # the advice is only actionable if it says which word to add.
        if not match.matched or not match.alias_skills:
            continue
        wanted = ", ".join(match.alias_skills)
        out.append(
            Suggestion(
                requirement_text=match.requirement.text,
                message=(
                    f"Matched on {wanted} only through a synonym. The posting's own "
                    f"wording does not appear on your resume — a keyword filter "
                    f"screening on it would score this as a miss."
                ),
                severity=match.requirement.importance,
            )
        )
    return out


def _gap_suggestions(matches: list[RequirementMatch]) -> list[Suggestion]:
    out: list[Suggestion] = []
    for match in matches:
        # Unscoreable requirements get their own "check by hand" list. Repeating
        # them here as advice would double-count a line the tool never assessed.
        if match.matched or not match.scoreable:
            continue
        requirement = match.requirement
        wanted = ", ".join(match.missing_skills or requirement.skills)

        if match.literal_skills or match.alias_skills:
            # Partly met: the posting asks for several things together and the
            # resume covers some of them. Naming the whole list here would send
            # someone off to add what they already have.
            have = ", ".join(match.literal_skills + match.alias_skills)
            message = (
                f"Asks for all of these together. You evidence {have}, but {wanted} "
                f"is missing — the requirement does not count until it is there."
            )
        else:
            message = (
                f"No evidence of {wanted} anywhere on the resume. If you have "
                f"it, it needs a line; if you do not, this is a real gap."
            )

        out.append(
            Suggestion(
                requirement_text=requirement.text,
                message=message,
                severity=requirement.importance,
            )
        )
    return out


def _quantification_suggestions(matches: list[RequirementMatch]) -> list[Suggestion]:
    out: list[Suggestion] = []
    for match in matches:
        if not match.matched or not match.evidence:
            continue
        if any(item.is_quantified for item in match.evidence):
            continue
        out.append(
            Suggestion(
                requirement_text=match.requirement.text,
                message=(
                    f'Supported only by unquantified lines, e.g. "'
                    f'{match.evidence[0].text[:60]}". A number here would carry '
                    f"more weight than the claim does on its own."
                ),
                severity=Importance.PREFERRED,
            )
        )
    return out


def _years_suggestions(matches: list[RequirementMatch]) -> list[Suggestion]:
    out: list[Suggestion] = []
    for match in matches:
        years = match.requirement.years
        if years is None:
            continue
        out.append(
            Suggestion(
                requirement_text=match.requirement.text,
                message=(
                    f"Asks for {years}+ years. This tool cannot verify tenure from "
                    f"a resume — confirm your dates make the case plainly."
                ),
                severity=match.requirement.importance,
            )
        )
    return out


def _semantic_suggestions(matches: list[RequirementMatch]) -> list[Suggestion]:
    """A model's judgement is worth flagging as a model's judgement."""
    out: list[Suggestion] = []
    for match in matches:
        if match.method is not MatchMethod.SEMANTIC:
            continue
        out.append(
            Suggestion(
                requirement_text=match.requirement.text,
                message=(
                    f"Counted as met by the semantic pass, not by anything the "
                    f"posting and your resume literally share. Worth a look: "
                    f'"{match.evidence[0].text[:58]}" — {match.note or ""}'
                ),
                severity=Importance.PREFERRED,
            )
        )
    return out


def build_report(
    job: JobSpec,
    resume: Resume,
    *,
    provider: str = "fixture",
    llm: LLMClient | None = None,
    max_semantic_calls: int | None = None,
) -> MatchReport:
    """Match, then derive advice from what the match established.

    Passing `llm` runs the semantic pass over whatever literal and alias
    matching could not reach. Without it the report is fully deterministic,
    which is the default and the mode the tests run in.
    """
    matches = matching.match_all(job, resume)

    if llm is not None:
        from resumatch.agents import resolve_semantically

        matches, _ = resolve_semantically(
            matches, resume, llm, max_calls=max_semantic_calls
        )
        provider = llm.name

    suggestions: list[Suggestion] = []
    suggestions.extend(_gap_suggestions(matches))
    suggestions.extend(_alias_suggestions(matches))
    suggestions.extend(_years_suggestions(matches))
    suggestions.extend(_quantification_suggestions(matches))
    suggestions.extend(_semantic_suggestions(matches))

    report = MatchReport(
        job_title=job.title,
        matches=matches,
        suggestions=suggestions,
        provider=provider,
    )

    # Whole-resume observations, appended after the per-requirement ones.
    noise = matching.unused_skills(job, resume)
    if len(noise) >= NOISE_THRESHOLD:
        report.suggestions.append(
            Suggestion(
                requirement_text="(whole resume)",
                message=(
                    f"{len(noise)} skills on your resume are never mentioned in this "
                    f"posting ({', '.join(noise[:5])}…). Not wrong, but they compete "
                    f"for attention with the ones being screened for."
                ),
                severity=Importance.PREFERRED,
            )
        )

    if report.required_score < POOR_FIT_THRESHOLD:
        report.suggestions.insert(
            0,
            Suggestion(
                requirement_text="(overall)",
                message=(
                    f"Only {report.required_score:.0%} of the required items are "
                    f"evidenced. That is a fit problem rather than a wording "
                    f"problem — editing individual lines will not close it."
                ),
                severity=Importance.REQUIRED,
            ),
        )

    return report


def format_report(report: MatchReport, *, verbose: bool = False) -> str:
    """Plain-text rendering, for the CLI and for pasting into a message."""
    lines: list[str] = []
    add = lines.append

    add("")
    add(f"  {report.job_title}")
    add(f"  {'-' * max(len(report.job_title), 20)}")
    add("")
    add(
        f"  Overall match      {report.score:.0%}   "
        f"({report.earned_weight}/{report.total_weight} weighted)"
    )
    add(f"  Required covered   {report.required_score:.0%}")
    add("")

    coverage = matching.coverage_by_importance(report.matches)
    for importance, (hit, total) in coverage.items():
        if total:
            add(f"    {importance.value:<10} {hit}/{total}")

    methods = matching.coverage_by_method(report.matches)
    if methods:
        add("")
        add("    matched by  " + "  ".join(f"{k}={v}" for k, v in methods.items()))
    add("")

    gaps = report.gaps(Importance.REQUIRED)
    if gaps:
        add("  Missing, required")
        for match in gaps:
            add(f"    x  {match.requirement.text[:68]}")
        add("")

    optional_gaps = report.gaps(Importance.PREFERRED)
    if optional_gaps:
        add("  Missing, nice to have")
        for match in optional_gaps:
            add(f"    -  {match.requirement.text[:68]}")
        add("")

    if verbose:
        add("  Matched")
        for match in report.hits():
            add(f"    v  [{match.method.value:<6}] {match.requirement.text[:56]}")
            for item in match.evidence[:1]:
                add(f"           {item.text[:64]}")
        add("")

    if report.suggestions:
        add("  What to change")
        for index, suggestion in enumerate(report.suggestions, start=1):
            marker = "!" if suggestion.severity is Importance.REQUIRED else "."
            add(f"    {marker} {suggestion.message}")
            if suggestion.requirement_text not in ("(overall)", "(whole resume)"):
                add(f"        re: {suggestion.requirement_text[:62]}")
            if index >= 12:
                remaining = len(report.suggestions) - index
                if remaining:
                    add(f"    … {remaining} more")
                break
        add("")

    return "\n".join(lines)
