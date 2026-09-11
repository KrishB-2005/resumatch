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

from dataclasses import dataclass

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


def _shares_spelling(requirement: Requirement, item: Evidence, skill: str) -> bool:
    """Whether posting and resume use the *same word* for one shared skill.

    Distinguishes "the posting said Python and so did the resume" from "the
    posting said k8s and the resume said Kubernetes". Both are matches; only
    the second needs the alias table to be believed, and only the second is at
    risk from a keyword filter.

    The comparison is per canonical skill. Comparing the whole requirement
    sentence against a resume line is never true and silently reports every
    match as an alias hit.

    Any spelling will do, as long as it is the *same* one on both sides: both
    documents saying "Postgres" is a literal hit even though the canonical name
    is "postgresql" and appears in neither.
    """
    wanted = skills.spelling_used(requirement.text, skill)
    return wanted is not None and wanted == skills.spelling_used(item.text, skill)


def is_conjunctive(text: str) -> bool:
    """Whether a multi-skill requirement wants all of them or any of them.

    "Docker and Kubernetes" asks for both. "React or TypeScript" asks for
    either. Scoring them the same way gives a resume with only Docker full
    marks on a requirement it half meets, which is the kind of quietly wrong
    number this project exists to avoid.

    The test is deliberately crude: the requirement has to actually say "and",
    and must not say "or". Everything else reads as the lenient case, because
    overstating a gap sends someone off to fix something that is not broken —
    "server-side JavaScript with Node.js" names two skills but is asking for
    one thing.
    """
    padded = f" {skills.normalise(text)} "
    return " and " in padded and " or " not in padded


@dataclass
class _Resolution:
    """What the resume had to say about one skill the posting named."""

    skill: str
    evidence: list[Evidence]
    literal: bool

    @property
    def found(self) -> bool:
        return bool(self.evidence)


def _resolve(requirement: Requirement, resume: Resume) -> list[_Resolution]:
    out: list[_Resolution] = []
    for skill in requirement.skills:
        evidence = _evidence_for_skill(resume, skill)
        out.append(
            _Resolution(
                skill=skill,
                evidence=evidence,
                literal=any(
                    _shares_spelling(requirement, item, skill) for item in evidence
                ),
            )
        )
    return out


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

    resolutions = _resolve(requirement, resume)
    found = [r for r in resolutions if r.found]
    absent = [r.skill for r in resolutions if not r.found]

    needs_all = len(resolutions) > 1 and is_conjunctive(requirement.text)
    satisfied = not absent if needs_all else bool(found)

    literal_skills = [r.skill for r in found if r.literal]
    alias_skills = [r.skill for r in found if not r.literal]

    if not satisfied:
        if needs_all and found:
            note = (
                f"asks for {' and '.join(r.skill for r in resolutions)}; "
                f"only {', '.join(r.skill for r in found)} is evidenced"
            )
        else:
            note = f"no evidence of {', '.join(requirement.skills)}"
        return RequirementMatch(
            requirement=requirement,
            method=MatchMethod.NONE,
            note=note,
            alias_skills=alias_skills,
            missing_skills=absent,
        )

    # Deduplicate while keeping resume order, then prefer quantified lines —
    # they are the better thing to show a reader.
    seen: set[str] = set()
    unique: list[Evidence] = []
    for resolution in found:
        for item in resolution.evidence:
            if item.text not in seen:
                seen.add(item.text)
                unique.append(item)
    unique.sort(key=lambda e: (not e.is_quantified,))

    # The weakest link decides the method. A requirement matched on Docker
    # literally and Kubernetes only through "k8s" is still one an automated
    # keyword filter can drop, so calling the whole thing EXACT would hide the
    # single most useful finding the tool produces.
    return RequirementMatch(
        requirement=requirement,
        method=MatchMethod.ALIAS if alias_skills else MatchMethod.EXACT,
        evidence=unique[:MAX_EVIDENCE],
        note=None
        if not alias_skills
        else (
            f"{', '.join(alias_skills)} matched through the alias table, "
            f"not the posting's own wording"
        ),
        literal_skills=literal_skills,
        alias_skills=alias_skills,
        missing_skills=absent,
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


def coverage_by_method(matches: list[RequirementMatch]) -> dict[str, int]:
    """How each match was made — literal, alias, or a model's judgement.

    Worth surfacing: "8 matched" reads very differently when six of them are
    exact and two are a model's opinion.
    """
    out: dict[str, int] = {}
    for match in matches:
        if not match.matched:
            continue
        out[match.method.value] = out.get(match.method.value, 0) + 1
    return out
