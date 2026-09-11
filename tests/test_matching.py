"""Coverage for the deterministic half — which is the whole thing, so far.

Everything here runs without a model, because the matcher is not allowed to
need one. That is the property that makes a score worth printing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resumatch import matching, skills
from resumatch.models import (
    Evidence,
    Importance,
    JobSpec,
    MatchMethod,
    Requirement,
    Resume,
    ResumeSection,
)
from resumatch.parsing import parse_job, parse_resume, read_text
from resumatch.report import build_report

SAMPLES = Path("data/samples")


def _resume(*lines: str, section: ResumeSection = ResumeSection.EXPERIENCE) -> Resume:
    return Resume(
        evidence=[
            Evidence(text=line, section=section, skills=skills.extract_skills(line))
            for line in lines
        ]
    )


def _job(*requirements: Requirement) -> JobSpec:
    return JobSpec(title="Test Role", requirements=list(requirements))


def _requirement(text: str, importance: Importance = Importance.REQUIRED) -> Requirement:
    return Requirement(
        text=text, importance=importance, skills=skills.extract_skills(text)
    )


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------


class TestSkillExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Strong Python skills", "python"),
            ("Comfortable with JS", "javascript"),
            ("Deploys on k8s", "kubernetes"),
            ("Uses sklearn daily", "scikit-learn"),
            ("Postgres experience", "postgresql"),
            ("Built with GPT-4", "openai"),
            ("Worked with Claude", "anthropic"),
        ],
    )
    def test_aliases_resolve_to_canonical(self, text: str, expected: str) -> None:
        assert expected in skills.extract_skills(text)

    def test_longer_phrases_win_over_the_words_inside_them(self) -> None:
        """"machine learning" must not also register some bare "learning"."""
        found = skills.extract_skills("machine learning and deep learning")
        assert "machine learning" in found
        assert "deep learning" in found

    def test_punctuated_skills_survive_normalisation(self) -> None:
        found = skills.extract_skills("C++ and C# and Node.js")
        assert {"c++", "c#", "node.js"} <= set(found)

    def test_a_skill_is_not_found_inside_an_unrelated_word(self) -> None:
        # "go" must not match "going", "ago", "golf".
        assert "go" not in skills.extract_skills("going to the golf course, an hour ago")

    def test_extraction_is_order_stable(self) -> None:
        text = "Python, Docker, Python again"
        assert skills.extract_skills(text) == skills.extract_skills(text)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_same_word_on_both_sides_is_an_exact_match(self) -> None:
        match = matching.match_requirement(
            _requirement("Strong Python"), _resume("Built a Python service")
        )
        assert match.method is MatchMethod.EXACT

    def test_different_words_for_the_same_skill_is_an_alias_match(self) -> None:
        """The finding the whole tool exists to surface."""
        match = matching.match_requirement(
            _requirement("Must know Kubernetes"), _resume("Ran services on k8s")
        )
        assert match.method is MatchMethod.ALIAS
        assert match.matched

    def test_a_shared_alias_spelling_still_counts_as_exact(self) -> None:
        """Both saying "Postgres" is literal, even though canonical is postgresql."""
        match = matching.match_requirement(
            _requirement("Postgres tuning"), _resume("Optimised Postgres queries")
        )
        assert match.method is MatchMethod.EXACT

    def test_absent_skill_is_not_matched(self) -> None:
        match = matching.match_requirement(
            _requirement("Terraform required"), _resume("Built a Python service")
        )
        assert not match.matched
        assert "terraform" in (match.note or "")

    def test_quantified_evidence_is_shown_first(self) -> None:
        match = matching.match_requirement(
            _requirement("Python"),
            _resume("Wrote Python tooling", "Cut Python build time by 40%"),
        )
        assert match.evidence[0].is_quantified

    def test_prose_requirement_is_unmatched_and_says_why(self) -> None:
        match = matching.match_requirement(
            _requirement("Partner with research teams"), _resume("Built things")
        )
        assert not match.matched
        assert "semantic pass" in (match.note or "")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_required_gaps_cost_twice_what_optional_ones_do(self) -> None:
        job = _job(
            _requirement("Python", Importance.REQUIRED),
            _requirement("Terraform", Importance.PREFERRED),
        )
        report = build_report(job, _resume("Python work"))
        # 2 of 3 weighted: the required hit earns 2, the optional miss loses 1.
        assert report.earned_weight == 2
        assert report.total_weight == 3

    def test_unscoreable_requirements_stay_out_of_the_denominator(self) -> None:
        """A requirement naming no skill was never assessed, so it cannot fail."""
        job = _job(
            _requirement("Python"),
            _requirement("4+ years of professional experience"),
        )
        report = build_report(job, _resume("Python work"))
        assert len(report.unscoreable) == 1
        assert report.score == 1.0, "an unassessable line must not deflate the score"

    def test_a_perfect_match_scores_one(self) -> None:
        job = _job(_requirement("Python"), _requirement("Docker"))
        report = build_report(job, _resume("Python and Docker in production"))
        assert report.score == 1.0
        assert report.required_score == 1.0

    def test_an_empty_posting_scores_zero_rather_than_dividing_by_zero(self) -> None:
        report = build_report(_job(), _resume("Python"))
        assert report.score == 0.0
        assert report.required_score == 0.0

    def test_every_point_traces_to_a_requirement(self) -> None:
        job = _job(_requirement("Python"), _requirement("Docker"))
        report = build_report(job, _resume("Python only"))
        assert report.earned_weight == sum(m.earned for m in report.hits())


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------


class TestAdvice:
    def test_an_alias_match_warns_about_keyword_filters(self) -> None:
        job = _job(_requirement("Kubernetes experience"))
        report = build_report(job, _resume("Ran services on k8s"))
        assert any("keyword filter" in s.message for s in report.suggestions)

    def test_an_exact_match_does_not_warn(self) -> None:
        job = _job(_requirement("Kubernetes experience"))
        report = build_report(job, _resume("Ran Kubernetes in production"))
        assert not any("keyword filter" in s.message for s in report.suggestions)

    def test_a_missing_skill_is_named_in_the_advice(self) -> None:
        job = _job(_requirement("Terraform required"))
        report = build_report(job, _resume("Python work"))
        assert any("terraform" in s.message for s in report.suggestions)

    def test_unquantified_support_is_flagged(self) -> None:
        job = _job(_requirement("Python"))
        report = build_report(job, _resume("Wrote some Python"))
        assert any("number here" in s.message for s in report.suggestions)

    def test_a_poor_fit_says_so_before_suggesting_line_edits(self) -> None:
        job = _job(
            _requirement("Terraform"), _requirement("Rust"), _requirement("Kafka")
        )
        report = build_report(job, _resume("Python work"))
        assert report.suggestions[0].message.startswith("Only ")
        assert "fit problem" in report.suggestions[0].message


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_resume_sections_are_detected(self) -> None:
        resume = parse_resume(
            "EXPERIENCE\nBuilt a Python service\nEDUCATION\nB.S. Computer Science"
        )
        sections = {item.section for item in resume.evidence}
        assert ResumeSection.EXPERIENCE in sections
        assert ResumeSection.EDUCATION in sections

    def test_preferred_heading_downgrades_everything_under_it(self) -> None:
        job = parse_job(
            "Requirements\n- Python\nNice to have\n- Terraform\n- Rust"
        )
        preferred = [r for r in job.requirements if r.importance is Importance.PREFERRED]
        assert {r.text for r in preferred} == {"Terraform", "Rust"}

    def test_company_prose_is_not_treated_as_a_requirement(self) -> None:
        """Marketing copy is not something a candidate can satisfy."""
        job = parse_job(
            "About us\nWe are a fast-growing team that loves Python.\n"
            "Requirements\n- Docker"
        )
        assert [r.text for r in job.requirements] == ["Docker"]

    def test_years_are_extracted(self) -> None:
        job = parse_job("Requirements\n- 5+ years of backend experience")
        assert job.requirements[0].years == 5

    def test_bullets_of_any_flavour_are_stripped(self) -> None:
        job = parse_job("Requirements\n• Python\n- Docker\n* Redis")
        assert [r.text for r in job.requirements] == ["Python", "Docker", "Redis"]


# ---------------------------------------------------------------------------
# End to end, on the bundled samples
# ---------------------------------------------------------------------------

requires_samples = pytest.mark.skipif(
    not (SAMPLES / "resume.txt").exists(), reason="bundled samples not present"
)


@requires_samples
class TestEndToEnd:
    def _report(self):
        return build_report(
            parse_job(read_text(SAMPLES / "job.txt")),
            parse_resume(read_text(SAMPLES / "resume.txt")),
        )

    def test_the_sample_pair_produces_a_usable_report(self) -> None:
        report = self._report()
        assert 0.0 < report.score < 1.0, "a partial match is the interesting case"
        assert report.matches
        assert report.suggestions

    def test_known_gaps_are_reported(self) -> None:
        """The sample resume has no LLM or Terraform work; the posting wants both."""
        report = self._report()
        missing = {s for m in report.gaps() for s in m.requirement.skills}
        assert "terraform" in missing
        assert "langchain" in missing

    def test_known_strengths_are_matched(self) -> None:
        report = self._report()
        matched = {s for m in report.hits() for s in m.requirement.skills}
        assert {"python", "docker", "kubernetes", "postgresql"} <= matched

    def test_the_report_is_deterministic(self) -> None:
        first, second = self._report(), self._report()
        assert first.score == second.score
        assert [s.message for s in first.suggestions] == [
            s.message for s in second.suggestions
        ]
