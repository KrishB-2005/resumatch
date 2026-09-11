"""Typed contracts for the matching pipeline.

The design decision this file encodes: a match is a *claim about a specific
requirement*, not a number. "78% match" on its own is unfalsifiable — you
cannot check it, argue with it, or act on it. Every point of the score here
traces back to a named requirement and the evidence that satisfied it, which
is what makes the result testable and what makes the advice specific.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Reject unknown keys rather than silently dropping them."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Importance(StrEnum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class RequirementKind(StrEnum):
    SKILL = "skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    RESPONSIBILITY = "responsibility"


class MatchMethod(StrEnum):
    """How a requirement was satisfied — cheapest and most defensible first."""

    EXACT = "exact"
    ALIAS = "alias"
    SEMANTIC = "semantic"
    NONE = "none"


# --------------------------------------------------------------------------
# Job description
# --------------------------------------------------------------------------


class Requirement(StrictModel):
    """One checkable thing the posting asks for."""

    text: str = Field(description="The requirement as written in the posting")
    kind: RequirementKind = RequirementKind.SKILL
    importance: Importance = Importance.REQUIRED
    skills: list[str] = Field(
        default_factory=list,
        description="Canonical skill tokens this requirement depends on",
    )
    years: int | None = Field(
        default=None, description="Years of experience asked for, when stated"
    )

    @property
    def weight(self) -> int:
        # A missing required skill costs twice what a missing nice-to-have does.
        return 2 if self.importance is Importance.REQUIRED else 1


class JobSpec(StrictModel):
    title: str = ""
    company: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    raw_text: str = ""

    @property
    def required(self) -> list[Requirement]:
        return [r for r in self.requirements if r.importance is Importance.REQUIRED]


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


class ResumeSection(StrEnum):
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    EDUCATION = "education"
    SKILLS = "skills"
    OTHER = "other"


class Evidence(StrictModel):
    """One line of the resume, with the skills it demonstrates."""

    text: str
    section: ResumeSection = ResumeSection.OTHER
    skills: list[str] = Field(default_factory=list)

    @property
    def is_quantified(self) -> bool:
        """Whether the line carries a number.

        Used only for advice, never for scoring — "improved throughput 5x"
        argues for itself in a way "improved throughput" does not.
        """
        return any(ch.isdigit() for ch in self.text)


class Resume(StrictModel):
    name: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    raw_text: str = ""

    def skills(self) -> set[str]:
        found: set[str] = set()
        for item in self.evidence:
            found.update(item.skills)
        return found


# --------------------------------------------------------------------------
# The match
# --------------------------------------------------------------------------


class RequirementMatch(StrictModel):
    requirement: Requirement
    method: MatchMethod = MatchMethod.NONE
    evidence: list[Evidence] = Field(default_factory=list)
    note: str | None = Field(
        default=None, description="Why this was or was not considered a match"
    )

    @property
    def matched(self) -> bool:
        return self.method is not MatchMethod.NONE

    @property
    def scoreable(self) -> bool:
        """Whether a deterministic verdict was even possible.

        "4+ years of backend experience" names no skill, so the matcher has
        nothing to check it against. Counting that as a miss would deflate the
        score with something the tool never actually assessed — it is reported
        for a human to check instead.
        """
        return bool(self.requirement.skills)

    @property
    def earned(self) -> int:
        return self.requirement.weight if self.matched else 0


class Suggestion(StrictModel):
    """An actionable change, tied to the gap it closes."""

    requirement_text: str
    message: str
    severity: Importance = Importance.REQUIRED


class MatchReport(StrictModel):
    job_title: str = ""
    matches: list[RequirementMatch] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    provider: str = "fixture"

    # -- scoring ---------------------------------------------------------

    @property
    def scored(self) -> list[RequirementMatch]:
        return [m for m in self.matches if m.scoreable]

    @property
    def unscoreable(self) -> list[RequirementMatch]:
        """Requirements with no named skill — listed, never scored."""
        return [m for m in self.matches if not m.scoreable]

    @property
    def total_weight(self) -> int:
        return sum(m.requirement.weight for m in self.scored)

    @property
    def earned_weight(self) -> int:
        return sum(m.earned for m in self.scored)

    @property
    def score(self) -> float:
        """Weighted coverage in [0, 1].

        Deliberately not a model's opinion: it is earned weight over total
        weight, so every point is attributable to a specific requirement.
        """
        return self.earned_weight / self.total_weight if self.total_weight else 0.0

    @property
    def required_score(self) -> float:
        """Coverage of the required items alone — the number that matters most."""
        required = [
            m for m in self.scored if m.requirement.importance is Importance.REQUIRED
        ]
        if not required:
            return 0.0
        return sum(1 for m in required if m.matched) / len(required)

    def gaps(self, importance: Importance | None = None) -> list[RequirementMatch]:
        out = [m for m in self.scored if not m.matched]
        if importance is not None:
            out = [m for m in out if m.requirement.importance is importance]
        return out

    def hits(self) -> list[RequirementMatch]:
        return [m for m in self.matches if m.matched]
