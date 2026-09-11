"""Deterministic requirement matching.

No model calls here. Given the same resume and posting this always produces the
same report, which is what makes the score testable and what lets the semantic
pass be an *addition* rather than the whole system.

The split matters. An LLM asked "how good a match is this?" returns a number
nobody can check. Asking it only "do these two specific phrases mean the same
skill?" — after literal and alias matching have already claimed everything they
can — is a question it is good at and whose answer is auditable.
"""

from __future__ import annotations

from resumatch import skills
from resumatch.models import (
    Evidence,
    Importance,
    JobSpec,
    MatchMethod,
    Requirement,
    RequirementMatch,
    Resume,
)

# How many supporting lines to attach to a match. More than a few is noise in
# the report and does not make the match any more true.
MAX_EVIDENCE = 3


def _evidence_for_skill(resume: Resume, skill: str) -> list[Evidence]:
    return [item for item in resume.evidence if skill in item.skills]


def _literal_hit(requirement: Requirement, item: Evidence) -> bool:
    """Whether posting and resume use the *same word* for a shared skill.

    Distinguishes "the posting said Python and so did the resume" from "the
    posting said k8s and the resume said Kubernetes". Both are matches; only
    the second needs the alias table to be believed, and only the second is at
    risk from a keyword filter.

    The comparison is per canonical skill. Comparing the whole requirement
    sentence against a resume line is never true and silently reports every
    match as an alias hit.
    """
    requirement_text = skills.normalise(requirement.text)
    evidence_text = skills.normalise(item.text)
    for skill in requirement.skills:
        # Any spelling will do, as long as it is the *same* one on both sides.
        # Both documents saying "Postgres" is a literal hit even though the
        # canonical name is "postgresql" and appears in neither.
        for spelling in (skill, *skills.aliases_of(skill)):
            token = skills.normalise(spelling)
            if token and token in requirement_text and token in evidence_text:
                return True
    return False


def match_requirement(requirement: Requirement, resume: Resume) -> RequirementMatch:
    """Try to satisfy one requirement from the resume, literal evidence first."""
    if not requirement.skills:
        # Nothing canonical to match on — a prose responsibility like "mentor
        # junior engineers". The semantic pass is the only thing that can speak
        # to these, so leave it unmatched rather than guessing.
        return RequirementMatch(
            requirement=requirement,
            method=MatchMethod.NONE,
            note="no canonical skill to match on; needs the semantic pass",
        )

    supporting: list[Evidence] = []
    for skill in requirement.skills:
        supporting.extend(_evidence_for_skill(resume, skill))

    if not supporting:
        missing = ", ".join(requirement.skills)
        return RequirementMatch(
            requirement=requirement,
            method=MatchMethod.NONE,
            note=f"no evidence of {missing}",
        )

    # Deduplicate while keeping resume order, then prefer quantified lines —
    # they are the better thing to show a reader.
    seen: set[str] = set()
    unique: list[Evidence] = []
    for item in supporting:
        if item.text not in seen:
            seen.add(item.text)
            unique.append(item)
    unique.sort(key=lambda e: (not e.is_quantified,))

    literal = any(_literal_hit(requirement, item) for item in unique)
    return RequirementMatch(
        requirement=requirement,
        method=MatchMethod.EXACT if literal else MatchMethod.ALIAS,
        evidence=unique[:MAX_EVIDENCE],
        note=None
        if literal
        else "matched through the alias table, not the posting's own wording",
    )


def match_all(job: JobSpec, resume: Resume) -> list[RequirementMatch]:
    return [match_requirement(r, resume) for r in job.requirements]


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def missing_skills(job: JobSpec, resume: Resume) -> list[str]:
    """Canonical skills the posting asks for that the resume never evidences."""
    have = resume.skills()
    wanted: list[str] = []
    for requirement in job.requirements:
        for skill in requirement.skills:
            if skill not in have and skill not in wanted:
                wanted.append(skill)
    return wanted


def unused_skills(job: JobSpec, resume: Resume) -> list[str]:
    """Skills the resume evidences that the posting never asks for.

    Not a defect — but on a long resume being read for thirty seconds, these
    are the lines competing for attention with the ones that matter.
    """
    wanted = {s for r in job.requirements for s in r.skills}
    return sorted(resume.skills() - wanted)


def coverage_by_importance(
    matches: list[RequirementMatch],
) -> dict[Importance, tuple[int, int]]:
    """{importance: (matched, total)} — the breakdown behind the headline score.

    Counts only what was actually scoreable, so this reconciles with the
    headline percentage instead of quietly using a larger denominator.
    """
    out: dict[Importance, tuple[int, int]] = {}
    scoreable = [m for m in matches if m.scoreable]
    for importance in Importance:
        subset = [m for m in scoreable if m.requirement.importance is importance]
        out[importance] = (sum(1 for m in subset if m.matched), len(subset))
    return out
