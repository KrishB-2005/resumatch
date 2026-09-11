"""The semantic pass — the only part that calls a model.

It runs last, on requirements literal and alias matching could not reach:
prose responsibilities like "partner with research to take prototypes to
production", which name no skill and so have nothing for the deterministic
matcher to check.

Two constraints keep it honest.

The model is only ever shown a handful of pre-ranked resume lines and asked
whether *those specific lines* satisfy *this specific requirement*. It never
sees the score, never sees the other requirements, and is never asked how good
the candidate is. A yes is therefore attributable to named evidence a reader
can go and check.

And a semantic match is recorded as `MatchMethod.SEMANTIC`, so the report can
always separate "this was literally on the resume" from "a model thought this
counted". They are not the same claim and should not look the same.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from resumatch.llm.client import LLMClient, LLMError
from resumatch.models import (
    Evidence,
    MatchMethod,
    Requirement,
    RequirementMatch,
    Resume,
)
from resumatch.retrieval import DEFAULT_TOP_K, EvidenceIndex

# A verdict the model is less sure about than this is discarded. A wrong
# semantic match is worse than an unresolved one: it tells someone they are
# covered when they are not, and they stop working on the gap.
MIN_CONFIDENCE = 0.6


class SemanticVerdict(BaseModel):
    """What the model returns for one requirement. Deliberately small."""

    satisfied: bool = Field(
        description="Whether the numbered lines actually demonstrate this requirement"
    )
    evidence_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of the lines that support the verdict",
    )
    confidence: float = Field(
        description="0 to 1. How sure you are, given only the lines shown."
    )
    reasoning: str = Field(description="One sentence. Cite the line that decided it.")

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        """Clamp rather than constrain.

        `Field(ge=0, le=1)` would be the obvious way to write this, but Pydantic
        emits it as `minimum`/`maximum` in the JSON schema and OpenAI's strict
        structured-output mode rejects numeric constraints with a 400 — so the
        bound has to be enforced here instead, after decoding.
        """
        return min(1.0, max(0.0, value))


INSTRUCTIONS = """\
You decide whether a candidate's resume lines satisfy one requirement from a \
job posting.

You will be given the requirement and a short numbered list of resume lines \
that were pre-selected as the most relevant. Judge only those lines.

Rules:
- Answer only about the lines shown. Do not assume anything the candidate did \
not write down.
- Related is not the same as equivalent. "Built internal tools" does not \
satisfy "mentor junior engineers"; "ran weekly code reviews for three \
engineers" does.
- Cite the deciding line by its number in `evidence_indices`. A satisfied \
verdict with no cited line is not useful.
- Prefer a low-confidence no to a high-confidence maybe. Someone will act on \
this by deciding a gap is closed, and a wrong yes stops them fixing it.
"""


def _prompt(requirement: Requirement, candidates: list[Evidence]) -> str:
    lines = [f"Requirement ({requirement.importance.value}): {requirement.text}", ""]
    lines.append("Candidate resume lines:")
    for index, item in enumerate(candidates, start=1):
        lines.append(f"{index}. [{item.section.value}] {item.text}")
    return "\n".join(lines)


def resolve_semantically(
    matches: list[RequirementMatch],
    resume: Resume,
    llm: LLMClient,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_calls: int | None = None,
) -> tuple[list[RequirementMatch], int]:
    """Try to settle unmatched requirements. Returns the matches and call count.

    Only unmatched requirements are considered — anything literal or alias
    matching already claimed is left alone, because a model cannot improve on
    a match that is already exact and would only add cost and doubt.
    """
    index = EvidenceIndex(resume.evidence)
    calls = 0

    for match in matches:
        if match.matched:
            continue
        if max_calls is not None and calls >= max_calls:
            break

        candidates = index.candidates_for(match.requirement, top_k)
        if not candidates:
            match.note = "no resume line was close enough to be worth judging"
            continue

        try:
            verdict = llm.parse(
                instructions=INSTRUCTIONS,
                user=_prompt(match.requirement, candidates),
                schema=SemanticVerdict,
            )
            calls += 1
        except LLMError as exc:
            # A model failure downgrades one requirement to unresolved. It does
            # not abort the report — the deterministic half still stands.
            match.note = f"semantic pass failed: {exc}"
            continue

        # A verdict either way means the requirement was assessed, which is
        # what makes it count toward the score.
        match.semantically_assessed = True

        if not verdict.satisfied or verdict.confidence < MIN_CONFIDENCE:
            match.note = verdict.reasoning or match.note
            continue

        cited = [
            candidates[i - 1]
            for i in verdict.evidence_indices
            if 1 <= i <= len(candidates)
        ]
        if not cited:
            # Satisfied with nothing cited is exactly the unfalsifiable answer
            # this design exists to refuse.
            match.note = "model claimed a match but cited no line; not counted"
            continue

        match.method = MatchMethod.SEMANTIC
        match.evidence = cited
        match.note = verdict.reasoning

    return matches, calls
