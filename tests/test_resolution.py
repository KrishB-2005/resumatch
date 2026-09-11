"""Per-skill resolution: and/or, synonyms, plurals, and ambiguous names.

Every test here was written against a bug the first implementation had. They
are grouped by the wrong answer they prevent, because that is the thing worth
remembering about each one.
"""

from __future__ import annotations

import pytest

from resumatch import skills
from resumatch.matching import is_conjunctive, match_requirement
from resumatch.models import (
    Evidence,
    Importance,
    MatchMethod,
    Requirement,
    Resume,
    ResumeSection,
)


def resume_of(*lines: str) -> Resume:
    return Resume(
        evidence=[
            Evidence(
                text=line,
                section=ResumeSection.EXPERIENCE,
                skills=skills.extract_skills(line),
            )
            for line in lines
        ]
    )


def requirement(text: str, importance: Importance = Importance.REQUIRED) -> Requirement:
    return Requirement(
        text=text, importance=importance, skills=skills.extract_skills(text)
    )


# -- "and" is not "or" -----------------------------------------------------


class TestConjunction:
    """A posting asking for two things is not satisfied by one of them.

    The first implementation treated every multi-skill requirement as a choice,
    so a resume with Docker alone scored full marks against "Docker and
    Kubernetes" — a number that is simply wrong in the candidate's favour.
    """

    def test_and_needs_both(self) -> None:
        match = match_requirement(
            requirement("Hands-on with Docker and Kubernetes in production"),
            resume_of("Containerised the API with Docker"),
        )

        assert not match.matched
        assert match.missing_skills == ["kubernetes"]

    def test_and_is_satisfied_when_both_are_there(self) -> None:
        match = match_requirement(
            requirement("Hands-on with Docker and Kubernetes in production"),
            resume_of("Ran Docker images on a Kubernetes cluster"),
        )

        assert match.matched

    def test_or_needs_only_one(self) -> None:
        match = match_requirement(
            requirement("Front-end work in React or TypeScript"),
            resume_of("Built the dashboard in React"),
        )

        assert match.matched

    def test_a_partial_conjunction_says_what_is_missing_not_what_is_there(
        self,
    ) -> None:
        """Naming the whole list would send someone to add what they already have."""
        match = match_requirement(
            requirement("Hands-on with Docker and Kubernetes in production"),
            resume_of("Containerised the API with Docker"),
        )

        assert "only docker is evidenced" in (match.note or "")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Docker and Kubernetes", True),
            ("React or TypeScript", False),
            ("Python, Go, or Rust", False),
            ("Postgres and Redis in production", True),
        ],
    )
    def test_conjunction_is_read_off_the_wording(self, text: str, expected: bool) -> None:
        assert is_conjunctive(text) is expected

    def test_a_single_skill_requirement_is_unaffected(self) -> None:
        match = match_requirement(
            requirement("Strong Python, including async and typing"),
            resume_of("Built an ingestion pipeline in Python"),
        )

        assert match.matched


# -- the synonym warning, per skill ----------------------------------------


class TestAliasIsPerSkill:
    """The project's headline finding, and it used to be suppressed.

    "Docker and Kubernetes" matched Docker literally, which marked the whole
    requirement EXACT and swallowed the fact that Kubernetes was only reachable
    via "k8s" — the exact case a keyword filter drops.
    """

    def test_a_literal_hit_does_not_hide_a_synonym_on_another_skill(self) -> None:
        match = match_requirement(
            requirement("Hands-on with Docker and Kubernetes in production"),
            resume_of("Migrated six containerised services to k8s using Docker"),
        )

        assert match.matched
        assert match.method is MatchMethod.ALIAS
        assert match.alias_skills == ["kubernetes"]
        assert match.literal_skills == ["docker"]

    def test_the_warning_names_only_the_skill_at_risk(self) -> None:
        from resumatch.report import _alias_suggestions

        match = match_requirement(
            requirement("Hands-on with Docker and Kubernetes in production"),
            resume_of("Migrated six containerised services to k8s using Docker"),
        )
        message = _alias_suggestions([match])[0].message

        assert "kubernetes" in message
        assert "docker" not in message, "docker is spelled the same in both"

    def test_everything_literal_stays_exact(self) -> None:
        match = match_requirement(
            requirement("Hands-on with Docker and Kubernetes in production"),
            resume_of("Ran Docker images on a Kubernetes cluster"),
        )

        assert match.method is MatchMethod.EXACT
        assert match.alias_skills == []


# -- plurals ---------------------------------------------------------------


class TestPlurals:
    """Postings pluralise; the vocabulary holds the singular."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Exposure to vector databases", "vector database"),
            ("Owns several microservices", "microservices"),
            ("Runs Kubernetes clusters", "kubernetes"),
        ],
    )
    def test_a_plural_mention_is_still_the_skill(self, text: str, expected: str) -> None:
        assert expected in skills.extract_skills(text)

    def test_a_plural_requirement_matches_a_singular_resume_line(self) -> None:
        match = match_requirement(
            requirement("Exposure to vector databases"),
            resume_of("Served embeddings from a vector database"),
        )

        assert match.matched

    def test_short_names_do_not_get_a_plural(self) -> None:
        """"goes" is not Go, and this is why the rule has a length floor."""
        assert skills.extract_skills("he goes to the standup with Python") == ["python"]


# -- ambiguous names -------------------------------------------------------


class TestAmbiguousNames:
    """A phantom skill is worse than a missed one: it invents evidence.

    "go to production", "the rest of the team" and "R&D" are ordinary English.
    Scored as Go, REST and R, they let a resume match requirements it never
    spoke to.
    """

    @pytest.mark.parametrize(
        "prose",
        [
            "the rest of the team",
            "go to production every Friday",
            "R&D for the group",
            "react to incidents within ten minutes",
            "express the design clearly in writing",
            "spark innovation across the org",
            "a swift resolution to the outage",
            "the project went off the rails",
            "took the helm of the team",
        ],
    )
    def test_ordinary_english_is_not_a_skill(self, prose: str) -> None:
        assert skills.extract_skills(prose) == []

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Spark jobs in Scala", "spark"),
            ("Built the iOS app in Swift", "swift"),
            ("Shipped a Rails monolith", "rails"),
            ("Helm charts for every service", "helm"),
        ],
    )
    def test_the_same_word_as_a_tool_still_counts(self, text: str, expected: str) -> None:
        """The denial list has to be idioms, not the words themselves."""
        assert expected in skills.extract_skills(text)

    def test_a_denied_mention_does_not_end_the_scan(self) -> None:
        """One sentence can use the word both ways."""
        assert skills.extract_skills("react to incidents in our React app") == ["react"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Languages: Python, JS, TS, SQL, some Go", "go"),
            ("Designed a Flask REST API", "rest"),
            ("Shipped a dashboard in React and TS", "react"),
        ],
    )
    def test_a_corroborated_mention_counts(self, text: str, expected: str) -> None:
        assert expected in skills.extract_skills(text)

    def test_a_distinctive_alias_never_needs_corroboration(self) -> None:
        """"golang" is not an English word, so it stands on its own."""
        assert skills.extract_skills("golang") == ["go"]
        assert skills.extract_skills("wrote restful endpoints") == ["rest"]

    def test_a_phantom_skill_cannot_satisfy_a_requirement(self) -> None:
        match = match_requirement(
            requirement("Production experience with Go"),
            resume_of("Worked with the rest of the team to go to production"),
        )

        assert not match.matched
