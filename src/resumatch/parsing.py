"""Turn a resume and a job posting into the typed shapes the matcher wants.

Both are free text written by humans who were not thinking about parsers, so
everything here is heuristic and everything here is deterministic. When a
heuristic cannot decide, it says so — an unclassified line becomes OTHER rather
than being forced into a section it might not belong to.

This is the layer where a score goes quietly wrong. Everything downstream is
tested against clean sentences, so it will happily produce a confident number
from three requirements when the posting had sixteen. `parsebench.py` measures
this file against whole documents for exactly that reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from resumatch import skills
from resumatch.models import (
    Evidence,
    Importance,
    JobSpec,
    Requirement,
    RequirementKind,
    Resume,
    ResumeSection,
)

# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_text(path: str | Path) -> str:
    """Text from a .pdf, or from any plain-text file."""
    source = Path(path)
    if source.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(source))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return source.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------

# Bullet glyphs, and the numbered lists enterprise postings prefer. The number
# form insists on a trailing space so a bare year like "2019." is left alone.
_BULLET = re.compile(r"^[ \t]*(?:[•▪●·*\-–—]\s*|\d+[.)]\s+)")
_WHITESPACE = re.compile(r"\s+")

# Contact details are not evidence of anything. Left in, the phone number
# becomes a resume line, and "github.com/someone" hands the candidate a `git`
# skill they never claimed.
_CONTACT = re.compile(
    r"[\w.+-]+@[\w-]+\.\w{2,}"
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"
    r"|^(?:https?://)?(?:www\.)?(?:linkedin|github|gitlab)\.com/\S*$",
    re.I,
)


@dataclass(frozen=True)
class _Line:
    text: str
    bulleted: bool


def _lines(text: str) -> list[_Line]:
    """Split into logical lines: bullets stripped, wrapped lines rejoined.

    A bullet that wraps is one requirement, not two. Left split, the tail
    becomes a requirement of its own — "and backwards compatibility across
    client versions" — which names no skill, can never be satisfied, and drags
    the score down over something the posting only asked for once.

    A line continues the one above it when it carries no bullet of its own and
    is either indented under it or starts mid-sentence in lower case. A blank
    line always ends the run.
    """
    out: list[_Line] = []
    after_blank = True

    for raw in text.splitlines():
        if not raw.strip():
            after_blank = True
            continue

        marker = _BULLET.match(raw)
        body = raw[marker.end() :] if marker else raw
        collapsed = _WHITESPACE.sub(" ", body).strip()
        if not collapsed:
            after_blank = True
            continue

        indented = marker is None and raw[:1].isspace()
        continues = (
            marker is None
            and bool(out)
            and not after_blank
            and (indented or collapsed[0].islower())
        )
        if continues:
            out[-1] = _Line(f"{out[-1].text} {collapsed}", out[-1].bulleted)
        else:
            out.append(_Line(collapsed, marker is not None))
        after_blank = False

    return out


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------

_SECTION_PATTERNS: list[tuple[ResumeSection, re.Pattern[str]]] = [
    (ResumeSection.EXPERIENCE, re.compile(r"^(work\s+)?experience|employment|professional", re.I)),
    (ResumeSection.PROJECTS, re.compile(r"^projects?|personal work|portfolio", re.I)),
    (ResumeSection.EDUCATION, re.compile(r"^education|academic", re.I)),
    (ResumeSection.SKILLS, re.compile(r"^(technical\s+)?skills|technologies|toolkit", re.I)),
    (ResumeSection.SUMMARY, re.compile(r"^summary|objective|about|profile", re.I)),
]

# A heading is short, has no sentence-ending punctuation, and names a section.
_MAX_HEADING_WORDS = 4


def _heading_for(line: _Line) -> ResumeSection | None:
    # A bullet is an item. "• Experience with Kubernetes" is something the
    # candidate did, not the start of a new section.
    if line.bulleted:
        return None
    stripped = line.text.strip().rstrip(":")
    if not stripped or len(stripped.split()) > _MAX_HEADING_WORDS:
        return None
    if stripped.endswith("."):
        return None
    for section, pattern in _SECTION_PATTERNS:
        if pattern.match(stripped):
            return section
    return None


def parse_resume(text: str, name: str = "") -> Resume:
    """Split a resume into evidence lines tagged with section and skills."""
    section = ResumeSection.OTHER
    evidence: list[Evidence] = []

    for line in _lines(text):
        heading = _heading_for(line)
        if heading is not None:
            section = heading
            continue

        if _CONTACT.search(line.text):
            continue

        # A skills block is usually "Languages: Python, Java, SQL" — one line
        # carrying many skills, which is evidence of each of them. Elsewhere a
        # line with no skill is still kept, since the semantic pass may match a
        # responsibility against it.
        evidence.append(
            Evidence(
                text=line.text,
                section=section,
                skills=skills.extract_skills(line.text),
            )
        )

    return Resume(name=name, evidence=evidence, raw_text=text)


# --------------------------------------------------------------------------
# Job posting
# --------------------------------------------------------------------------

_PREFERRED_HEADING = re.compile(
    r"nice[\s-]*to[\s-]*have|preferred|bonus|plus(es)?\b|desirable|good to have", re.I
)
_REQUIRED_HEADING = re.compile(
    r"require|must[\s-]*have|qualification|minimum"
    r"|what you.ll need|what we.re looking for|who you are|about you"
    r"|what you bring|you.ll bring",
    re.I,
)
_RESPONSIBILITY_HEADING = re.compile(
    r"responsibilit|what you.ll do|day[\s-]*to[\s-]*day|^the role$|in this role", re.I
)
# Marketing prose about the company is not something a candidate can satisfy.
# Lines under these headings are skipped entirely rather than scored as
# requirements the resume will always "fail".
_CONTEXT_HEADING = re.compile(
    r"about (the role|us|the team|the company)|overview|who we are|our mission", re.I
)

# Postings write the number both ways — "8+ years", "Five or more years" — and
# a digits-only pattern silently reports no requirement at all for the second.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_YEARS = re.compile(
    r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\b"
    # "8+", "3-5", "five or more" — a range means the floor is what is asked.
    r"(?:\s*\+|\s*-\s*\d{1,2}|\s+or\s+more|\s+plus)?"
    r"\s*years?\b",
    re.I,
)


def _years_in(text: str) -> int | None:
    match = _YEARS.search(text)
    if match is None:
        return None
    token = match.group(1).lower()
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]
_TITLE = re.compile(r"^(?:title|role|position)\s*:\s*(.+)$", re.I)
_COMPANY = re.compile(r"^company\s*:\s*(.+)$", re.I)

_EDUCATION_HINT = re.compile(r"\b(bachelor|master|phd|b\.?s\.?|m\.?s\.?|degree)\b", re.I)

_MAX_JOB_HEADING_WORDS = 6


def _kind_for(line: str, in_responsibilities: bool) -> RequirementKind:
    if _EDUCATION_HINT.search(line):
        return RequirementKind.EDUCATION
    if _YEARS.search(line):
        return RequirementKind.EXPERIENCE
    if in_responsibilities:
        return RequirementKind.RESPONSIBILITY
    return RequirementKind.SKILL


def _has_heading_shape(line: _Line) -> bool:
    """Short, unbulleted, unpunctuated — the shape of a section label.

    Shape alone is never enough to discard a line. Plenty of postings write
    "Requirements" and then one item per line with no bullet, so "Python",
    "Five years of backend experience" and "Own the deployment pipeline" all
    have this shape while being the actual content of the posting.

    The bullet test is the one that matters here. Without it "• Strong Python
    required" is short and contains "required", so it reads as a *Requirements*
    heading and that requirement vanishes from the posting entirely.
    """
    if line.bulleted:
        return False
    if line.text.endswith(":"):
        return True
    return len(line.text.split()) <= _MAX_JOB_HEADING_WORDS and not line.text.endswith(
        (".", "!", "?")
    )


@dataclass
class _Bucket:
    importance: Importance = Importance.REQUIRED
    responsibilities: bool = False
    context: bool = False


_KNOWN_HEADINGS = (
    _CONTEXT_HEADING,
    _PREFERRED_HEADING,
    _RESPONSIBILITY_HEADING,
    _REQUIRED_HEADING,
)


def _names_a_section(line: _Line) -> bool:
    """Whether this line is a heading the parser actually recognises.

    Distinct from `_has_heading_shape`, which only judges shape. A job title,
    and every item in a bullet-less requirements list, has exactly the shape of
    a heading — so shape alone can neither mark where the masthead ends nor
    decide what to discard.
    """
    return _has_heading_shape(line) and any(
        pattern.search(line.text) for pattern in _KNOWN_HEADINGS
    )


def _switch(bucket: _Bucket, line: str) -> None:
    """Point the bucket at whatever section this recognised heading opens."""
    if _CONTEXT_HEADING.search(line):
        bucket.context, bucket.responsibilities = True, False
        return
    if _PREFERRED_HEADING.search(line):
        bucket.importance = Importance.PREFERRED
        bucket.responsibilities = bucket.context = False
        return
    if _RESPONSIBILITY_HEADING.search(line):
        bucket.importance = Importance.REQUIRED
        bucket.responsibilities, bucket.context = True, False
        return
    if _REQUIRED_HEADING.search(line):
        bucket.importance = Importance.REQUIRED
        bucket.responsibilities = bucket.context = False


def parse_job(text: str) -> JobSpec:
    """Split a posting into requirements, flagged required vs preferred.

    Importance comes from the heading a line sits under, because that is how
    postings actually encode it — a "Nice to have" block means every line in it
    is optional regardless of how it is phrased.
    """
    title = ""
    company = ""
    bucket = _Bucket()
    requirements: list[Requirement] = []

    lines = _lines(text)

    # Everything above the first recognised heading is the masthead: job title,
    # company, office, salary band. Scored as requirements, those are permanent
    # misses the candidate can do nothing about.
    first_heading = next(
        (i for i, line in enumerate(lines) if _names_a_section(line)), None
    )

    for index, line in enumerate(lines):
        match = _TITLE.match(line.text)
        if match:
            title = match.group(1).strip()
            continue
        match = _COMPANY.match(line.text)
        if match:
            company = match.group(1).strip()
            continue

        if _names_a_section(line):
            _switch(bucket, line.text)
            continue

        # An unrecognised heading still ends a context block. "Our mission"
        # runs until something else starts, and if "The role" does not clear it
        # then the whole posting is discarded as marketing copy. Only a line
        # shaped like a heading gets to do this, and it is not kept — outside a
        # context block the same line would be content.
        if bucket.context and _has_heading_shape(line):
            bucket.context = False
            continue

        if bucket.context or (first_heading is not None and index < first_heading):
            continue

        requirements.append(
            Requirement(
                text=line.text,
                kind=_kind_for(line.text, bucket.responsibilities),
                importance=bucket.importance,
                skills=skills.extract_skills(line.text),
                years=_years_in(line.text),
            )
        )

    if not title and lines:
        # Fall back to the first line, which is the headline in most postings
        # people paste in.
        title = lines[0].text

    return JobSpec(
        title=title, company=company, requirements=requirements, raw_text=text
    )
