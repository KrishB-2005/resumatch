"""The benchmark runs in CI, so a regression cannot be committed quietly.

A benchmark nobody runs is a file, not a measurement.
"""

from __future__ import annotations

from resumatch.benchmark import load_cases, run


def test_every_case_still_gets_the_labelled_answer() -> None:
    results = run()
    failures = [f"{r.case.id}: {r.failure}" for r in results if not r.passed]

    assert not failures, "\n".join(failures)


def test_the_benchmark_covers_the_cases_worth_covering() -> None:
    """Guards against the other failure mode: a benchmark quietly shrinking.

    Deleting the cases that fail is the easiest way to make a score go up, so
    the count has a floor and the interesting categories are named.
    """
    ids = {case.id for case in load_cases()}

    assert len(ids) >= 30
    for topic in ("alias", "conjunction", "ambiguous", "plural", "unscoreable"):
        assert any(topic in name for name in ids), f"no {topic} cases left"


def test_every_case_says_why_it_expects_what_it_expects() -> None:
    """A label without reasoning cannot be argued with when it fails."""
    for case in load_cases():
        assert case.why.strip(), f"{case.id} has no reasoning"


def test_unscoreable_cases_are_never_counted_as_matches() -> None:
    """The honesty property, asserted over the labelled set.

    A requirement the matcher cannot rule on must stay out of the score
    entirely rather than landing in it as a miss.
    """
    for result in run():
        if not result.case.scoreable:
            assert not result.match.scoreable
            assert not result.match.matched
