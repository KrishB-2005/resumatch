"""The parser, and the bugs running whole documents through it turned up.

Each class is named for the wrong answer it prevents. Every one was live until
real postings and resumes went through `parse_job`/`parse_resume` — the unit
tests passed the whole time, because they fed the parser clean sentences
somebody had already extracted by hand.
"""

from __future__ import annotations

import pytest

from resumatch.models import Importance, RequirementKind, ResumeSection
from resumatch.parsebench import load_cases, run
from resumatch.parsing import parse_job, parse_resume


def texts(requirements) -> str:
    return " || ".join(r.text for r in requirements)


class TestWrappedLines:
    """A bullet that wraps is one requirement, not two.

    The tail — "and backwards compatibility across client versions" — named no
    skill, could never be satisfied, and cost the candidate score for something
    the posting asked once.
    """

    def test_an_indented_continuation_joins_the_bullet_above(self) -> None:
        job = parse_job(
            "Requirements\n"
            "- Experience designing REST APIs at scale, including versioning\n"
            "  and backwards compatibility across client versions\n"
            "- Comfort with Postgres\n"
        )

        assert len(job.requirements) == 2
        assert "backwards compatibility" in job.requirements[0].text

    def test_a_lowercase_continuation_joins_even_unindented(self) -> None:
        job = parse_job(
            "Requirements\n"
            "- A track record of owning systems that handle money, where\n"
            "correctness is not negotiable\n"
        )

        assert len(job.requirements) == 1

    def test_a_blank_line_ends_the_run(self) -> None:
        job = parse_job("Requirements\n- Python\n\nkubernetes in production\n")

        assert len(job.requirements) == 2

    def test_resume_bullets_join_too(self) -> None:
        resume = parse_resume(
            "EXPERIENCE\n"
            "- Ran the migration from a single Postgres instance to a sharded\n"
            "  setup with no customer-visible downtime.\n"
        )

        assert len(resume.evidence) == 1
        assert "sharded setup" in resume.evidence[0].text


class TestBulletsAreNotHeadings:
    """"• Strong Python required" is a requirement, not a Requirements heading.

    It is short and contains the word "required", which was enough to make the
    parser treat it as a section label and drop it from the posting entirely.
    """

    @pytest.mark.parametrize(
        "bullet",
        [
            "• Strong Python required",
            "• PyTorch experience is a must",
            "- Kubernetes required in production",
            "1. Minimum three years of Python",
        ],
    )
    def test_a_bullet_naming_a_heading_word_survives(self, bullet: str) -> None:
        job = parse_job(f"REQUIREMENTS\n{bullet}\n")

        assert len(job.requirements) == 1

    def test_a_one_word_bullet_is_still_a_requirement(self) -> None:
        job = parse_job("REQUIREMENTS\n• SQL\n• Docker\n")

        assert len(job.requirements) == 2
        assert job.requirements[0].skills == ["sql"]

    def test_numbered_items_are_parsed_as_items(self) -> None:
        job = parse_job(
            "Minimum Qualifications\n"
            "1. Bachelor's degree in Computer Science\n"
            "2. 8+ years of professional experience\n"
            "3. Proficiency in Java or Scala\n"
        )

        assert len(job.requirements) == 3
        assert not job.requirements[0].text.startswith("1.")
        assert job.requirements[1].years == 8


class TestContextBlocksEnd:
    """"Our mission" must not swallow the rest of the posting.

    Nothing after a context heading was collected until another *recognised*
    heading appeared. A posting whose next heading was "The role" parsed to a
    single requirement — its own title — and scored the candidate against
    nothing at all.
    """

    def test_an_unrecognised_heading_ends_the_context_block(self) -> None:
        job = parse_job(
            "Full Stack Engineer\n\n"
            "Our mission\n"
            "We think hiring is broken and we are fixing it.\n\n"
            "The role\n"
            "You should be fluent in TypeScript with production Node.js.\n"
        )

        assert "TypeScript" in texts(job.requirements)
        assert "hiring is broken" not in texts(job.requirements)

    def test_a_posting_that_is_all_prose_still_yields_skills(self) -> None:
        job = parse_job(
            "Engineer\n\n"
            "About us\nWe are a startup.\n\n"
            "What we're looking for\n"
            "Four years of Python, and real experience with Postgres and AWS.\n"
        )

        found = {s for r in job.requirements for s in r.skills}
        assert {"python", "postgresql", "aws"} <= found


class TestMasthead:
    """The job title and the company line are not requirements.

    Scored as requirements they are permanent misses no candidate can fix, and
    they inflate the denominator.
    """

    def test_lines_above_the_first_heading_are_dropped(self) -> None:
        job = parse_job(
            "Senior Backend Engineer, Payments\n"
            "Lumen Financial - New York, NY - Hybrid\n\n"
            "What we're looking for\n"
            "- Deep Python experience\n"
        )

        assert len(job.requirements) == 1
        assert "Lumen Financial" not in texts(job.requirements)

    def test_the_headline_still_becomes_the_title(self) -> None:
        job = parse_job(
            "Senior Backend Engineer, Payments\n\nRequirements\n- Python\n"
        )

        assert "Senior Backend Engineer" in job.title

    def test_a_posting_with_no_headings_keeps_everything(self) -> None:
        """Dropping the masthead must not mean dropping the whole document."""
        job = parse_job("We need someone strong in Python and Docker.\n")

        assert job.requirements


class TestContactDetails:
    """A phone number is not evidence, and a GitHub URL is not a git skill."""

    def test_a_contact_line_is_not_evidence(self) -> None:
        resume = parse_resume(
            "Priya Raman\n"
            "priya.raman@example.com | (555) 018-2274 | github.com/praman\n\n"
            "EXPERIENCE\n"
            "- Rebuilt the ledger in Python\n"
        )

        assert "555" not in " ".join(e.text for e in resume.evidence)

    def test_a_profile_url_does_not_grant_the_git_skill(self) -> None:
        resume = parse_resume("Dana Whitfield\nlinkedin.com/in/dwhitfield\n")

        assert "git" not in resume.skills()

    def test_github_stars_in_a_real_bullet_are_kept(self) -> None:
        """Only contact lines go — a project bullet that says GitHub stays."""
        resume = parse_resume(
            "PROJECTS\n- Parses accounting files. 700 GitHub stars.\n"
        )

        assert len(resume.evidence) == 1


class TestWhitespace:
    """pypdf hands back doubled spaces; the report shows evidence verbatim."""

    def test_runs_of_whitespace_collapse(self) -> None:
        resume = parse_resume(
            "Experience\nBuilt  the  streaming  ingestion  layer  in  Scala\n"
        )

        assert resume.evidence[0].text == "Built the streaming ingestion layer in Scala"

    def test_skills_survive_the_doubled_spacing(self) -> None:
        resume = parse_resume("Skills\nScala, Python,  SQL,   Spark,  Kafka\n")

        assert {"scala", "python", "sql", "spark", "kafka"} <= resume.skills()


class TestYears:
    """"Five or more years" is a number too.

    A digits-only pattern reported no tenure requirement at all for every
    posting that spells it out, so the "this tool cannot verify tenure" warning
    never fired on them.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("8+ years of professional experience", 8),
            ("Five or more years building backend services", 5),
            ("Five years of backend experience", 5),
            ("ten years in the industry", 10),
            ("3-5 years of experience", 3),
            ("Strong Python skills", None),
            ("Graduated in 2019", None),
        ],
    )
    def test_tenure_is_read_in_words_or_digits(
        self, text: str, expected: int | None
    ) -> None:
        job = parse_job(f"Requirements\n- {text}\n")

        assert job.requirements[0].years == expected

    def test_a_range_asks_for_its_floor(self) -> None:
        job = parse_job("Requirements\n- 3-5 years of Python\n")

        assert job.requirements[0].years == 3


class TestSectionsAndKinds:
    def test_headings_assign_sections(self) -> None:
        resume = parse_resume(
            "SUMMARY\nBackend engineer.\n\n"
            "WORK EXPERIENCE\n- Built a ledger.\n\n"
            "PROJECTS\n- ledgerfmt, a formatter.\n\n"
            "EDUCATION\nB.S. Computer Science\n\n"
            "TECHNICAL SKILLS\nPython, Docker\n"
        )

        by_section = {e.section for e in resume.evidence}
        assert by_section == {
            ResumeSection.SUMMARY,
            ResumeSection.EXPERIENCE,
            ResumeSection.PROJECTS,
            ResumeSection.EDUCATION,
            ResumeSection.SKILLS,
        }

    def test_responsibility_headings_set_the_kind(self) -> None:
        job = parse_job("Responsibilities\n- Mentor two junior engineers\n")

        assert job.requirements[0].kind is RequirementKind.RESPONSIBILITY

    def test_preferred_headings_downgrade_what_follows(self) -> None:
        job = parse_job("Requirements\n- Python\n\nNice to haves\n- Terraform\n")

        assert job.requirements[0].importance is Importance.REQUIRED
        assert job.requirements[1].importance is Importance.PREFERRED


# -- the suite itself ------------------------------------------------------


def test_every_document_parses_as_labelled() -> None:
    failures = [
        f"{r.case.id}: {'; '.join(r.failures)}" for r in run() if not r.passed
    ]

    assert not failures, "\n".join(failures)


def test_the_corpus_covers_both_document_kinds_and_says_why() -> None:
    cases = load_cases()

    assert len({c.kind for c in cases}) == 2
    assert len(cases) >= 7
    for case in cases:
        assert case.why.strip(), f"{case.id} has no reasoning"
