"""The semantic pass and the retrieval that feeds it.

No network and no key. The provider is scripted, which is the whole point of
putting model access behind one method: the pass can be tested for the things
that actually matter — that it refuses bad answers, that it cannot silently
inflate a score, and that a failure degrades one requirement instead of the
whole report.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from resumatch.agents import MIN_CONFIDENCE, SemanticVerdict, resolve_semantically
from resumatch.llm.client import FixtureClient, LLMError, build_client
from resumatch.matching import match_all
from resumatch.models import (
    Evidence,
    Importance,
    JobSpec,
    MatchMethod,
    Requirement,
    RequirementKind,
    Resume,
    ResumeSection,
)
from resumatch.report import build_report
from resumatch.retrieval import MIN_SCORE, EvidenceIndex


class ScriptedClient:
    """Returns verdicts from a queue. Fails if asked more than it was given."""

    name = "scripted"

    def __init__(self, *verdicts: SemanticVerdict | Exception) -> None:
        self.queue = list(verdicts)
        self.prompts: list[str] = []
        self.calls = 0

    def parse(self, *, instructions: str, user: str, schema: type[BaseModel]):
        self.prompts.append(user)
        self.calls += 1
        if not self.queue:
            raise AssertionError("the pass made more calls than the test scripted")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def verdict(
    satisfied: bool = True,
    indices: list[int] | None = None,
    confidence: float = 0.9,
    reasoning: str = "line 1 says it plainly",
) -> SemanticVerdict:
    return SemanticVerdict(
        satisfied=satisfied,
        evidence_indices=[1] if indices is None else indices,
        confidence=confidence,
        reasoning=reasoning,
    )


@pytest.fixture
def resume() -> Resume:
    return Resume(
        name="Test Candidate",
        evidence=[
            Evidence(
                text=(
                    "Ran weekly code reviews for three junior engineers and "
                    "onboarded two of them"
                ),
                section=ResumeSection.EXPERIENCE,
            ),
            Evidence(
                text="Built an internal dashboard in React",
                section=ResumeSection.PROJECTS,
                skills=["react"],
            ),
            Evidence(
                text="Wrote the deployment runbook the on-call rotation uses",
                section=ResumeSection.EXPERIENCE,
            ),
        ],
    )


def mentoring() -> Requirement:
    """Names no canonical skill, so only the semantic pass can reach it."""
    return Requirement(
        text="Mentor junior engineers and grow the team's review culture",
        kind=RequirementKind.RESPONSIBILITY,
        importance=Importance.REQUIRED,
        skills=[],
    )


def job(*requirements: Requirement) -> JobSpec:
    return JobSpec(title="Test Role", requirements=list(requirements))


# -- retrieval -------------------------------------------------------------


def test_retrieval_ranks_the_relevant_line_first(resume: Resume) -> None:
    index = EvidenceIndex(resume.evidence)
    found = index.candidates_for(mentoring())

    assert found, "a shared word should have produced at least one candidate"
    assert "code reviews" in found[0].text


def test_retrieval_drops_lines_with_nothing_in_common(resume: Resume) -> None:
    index = EvidenceIndex(resume.evidence)
    found = index.candidates_for(
        Requirement(text="Kubernetes operator development in Go", skills=[])
    )

    assert found == []


def test_retrieval_ignores_filler_words(resume: Resume) -> None:
    """Otherwise every line matches "experience with" and the ranking is noise."""
    index = EvidenceIndex(resume.evidence)
    hits = index.search("experience with building and working on the")

    assert hits == []


def test_retrieval_respects_top_k(resume: Resume) -> None:
    index = EvidenceIndex(resume.evidence)
    hits = index.search("code reviews engineers dashboard react runbook", top_k=2)

    assert len(hits) <= 2
    assert all(h.score >= MIN_SCORE for h in hits)


# -- the pass itself -------------------------------------------------------


def test_a_cited_yes_becomes_a_semantic_match(resume: Resume) -> None:
    matches = match_all(job(mentoring()), resume)
    client = ScriptedClient(verdict())

    matches, calls = resolve_semantically(matches, resume, client)

    assert calls == 1
    assert matches[0].method is MatchMethod.SEMANTIC
    assert "code reviews" in matches[0].evidence[0].text
    assert matches[0].matched


def test_satisfied_with_no_citation_is_refused(resume: Resume) -> None:
    """The unfalsifiable answer this design exists to reject."""
    matches = match_all(job(mentoring()), resume)
    client = ScriptedClient(verdict(indices=[]))

    matches, _ = resolve_semantically(matches, resume, client)

    assert not matches[0].matched
    assert "cited no line" in (matches[0].note or "")


def test_out_of_range_citation_is_refused(resume: Resume) -> None:
    matches = match_all(job(mentoring()), resume)
    client = ScriptedClient(verdict(indices=[99]))

    matches, _ = resolve_semantically(matches, resume, client)

    assert not matches[0].matched


def test_low_confidence_yes_is_not_counted(resume: Resume) -> None:
    matches = match_all(job(mentoring()), resume)
    client = ScriptedClient(verdict(confidence=MIN_CONFIDENCE - 0.01))

    matches, _ = resolve_semantically(matches, resume, client)

    assert not matches[0].matched


def test_a_model_failure_degrades_one_requirement_not_the_report(
    resume: Resume,
) -> None:
    matches = match_all(job(mentoring(), mentoring()), resume)
    client = ScriptedClient(LLMError("rate limited"), verdict())

    matches, _ = resolve_semantically(matches, resume, client)

    assert "rate limited" in (matches[0].note or "")
    assert matches[1].method is MatchMethod.SEMANTIC


def test_already_matched_requirements_are_never_sent(resume: Resume) -> None:
    """Paying a model to second-guess an exact match buys doubt, nothing else."""
    spec = job(Requirement(text="React experience", skills=["react"]), mentoring())
    matches = match_all(spec, resume)
    assert matches[0].method is MatchMethod.EXACT

    client = ScriptedClient(verdict())
    matches, calls = resolve_semantically(matches, resume, client)

    assert calls == 1
    assert "React experience" not in client.prompts[0]


def test_max_calls_caps_spend(resume: Resume) -> None:
    matches = match_all(job(mentoring(), mentoring(), mentoring()), resume)
    client = ScriptedClient(verdict(), verdict())

    matches, calls = resolve_semantically(matches, resume, client, max_calls=1)

    assert calls == 1


def test_a_requirement_with_no_candidates_costs_nothing(resume: Resume) -> None:
    unreachable = Requirement(text="Kubernetes operator development in Go", skills=[])
    matches = match_all(job(unreachable), resume)
    client = ScriptedClient()

    matches, calls = resolve_semantically(matches, resume, client)

    assert calls == 0
    assert "close enough" in (matches[0].note or "")


def test_the_prompt_shows_only_the_shortlist(resume: Resume) -> None:
    matches = match_all(job(mentoring()), resume)
    client = ScriptedClient(verdict())

    resolve_semantically(matches, resume, client)

    prompt = client.prompts[0]
    assert "Mentor junior engineers" in prompt
    assert "1. [experience]" in prompt
    # The score, the other requirements and the candidate's name stay out of it.
    assert "Test Candidate" not in prompt
    assert "%" not in prompt


# -- scoring symmetry ------------------------------------------------------


def test_a_semantic_no_makes_the_requirement_count_as_a_miss(resume: Resume) -> None:
    """The pass must be able to lower a score, or it is not an assessment.

    Scoring only the yes verdicts would mean every run could only ever help,
    which is a score-inflation device dressed up as a judgement.
    """
    spec = job(mentoring())
    deterministic = build_report(spec, resume)
    assert deterministic.total_weight == 0, "unreachable without a model"

    client = ScriptedClient(verdict(satisfied=False, reasoning="no line shows it"))
    judged = build_report(spec, resume, llm=client)

    assert judged.total_weight == 2, "a verdict either way makes it scoreable"
    assert judged.earned_weight == 0
    assert judged.score == 0.0


def test_a_semantic_yes_earns_the_requirements_full_weight(resume: Resume) -> None:
    client = ScriptedClient(verdict())
    report = build_report(job(mentoring()), resume, llm=client)

    assert report.earned_weight == 2
    assert report.score == 1.0


def test_a_refused_verdict_is_still_assessed(resume: Resume) -> None:
    """Refusing the answer is not the same as never having asked."""
    client = ScriptedClient(verdict(indices=[]))
    report = build_report(job(mentoring()), resume, llm=client)

    assert report.total_weight == 2
    assert report.earned_weight == 0


def test_a_failed_call_leaves_the_requirement_unscoreable(resume: Resume) -> None:
    """No verdict means no assessment, so it must not count against the score."""
    client = ScriptedClient(LLMError("network down"))
    report = build_report(job(mentoring()), resume, llm=client)

    assert report.total_weight == 0
    assert report.unscoreable


def test_the_report_names_the_provider_that_judged_it(resume: Resume) -> None:
    report = build_report(job(mentoring()), resume, llm=ScriptedClient(verdict()))

    assert report.provider == "scripted"


def test_semantic_matches_are_flagged_as_model_judged(resume: Resume) -> None:
    report = build_report(job(mentoring()), resume, llm=ScriptedClient(verdict()))

    assert any("semantic pass" in s.message for s in report.suggestions)


# -- providers -------------------------------------------------------------


def test_the_fixture_provider_declines_rather_than_guesses(resume: Resume) -> None:
    """A stub that said yes would make the keyless path outscore the real one."""
    report = build_report(job(mentoring()), resume, llm=FixtureClient())

    assert report.earned_weight == 0


def test_an_unknown_provider_is_an_error_not_a_fallback() -> None:
    with pytest.raises(LLMError, match="unknown provider"):
        build_client("gpt5-turbo-max")


def test_the_default_provider_is_fixture_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RESUMATCH_PROVIDER", raising=False)

    assert build_client().name == "fixture"
