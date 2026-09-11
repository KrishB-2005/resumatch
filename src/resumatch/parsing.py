"""Turn a resume and a job posting into the typed shapes the matcher wants.

Both are free text written by humans who were not thinking about parsers, so
everything here is heuristic and everything here is deterministic. When a
heuristic cannot decide, it says so — an unclassified line becomes OTHER rather
than being forced into a section it might not belong to.
"""

from __future__ import annotations

import re
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

_BULLET = re.compile(r"^\s*[•▪●·*\-–—]\s*")


def _heading_for(line: str) -> ResumeSection | None:
    stripped = line.strip().rstrip(":")
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

    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue

        heading = _heading_for(line)
        if heading is not None:
            section = heading
            continue

        # A skills block is usually "Languages: Python, Java, SQL" — one line
        # carrying many skills, which is evidence of each of them.
        found = skills.extract_skills(line)
        if not found and section is not ResumeSection.SKILLS:
            # No skill and not in the skills block: still keep it, since the
            # semantic pass may match a responsibility against it.
            evidence.append(Evidence(text=line, section=section, skills=[]))
            continue

        evidence.append(Evidence(text=line, section=section, skills=found))

    return Resume(name=name, evidence=evidence, raw_text=text)


# --------------------------------------------------------------------------
# Job posting
# --------------------------------------------------------------------------

_PREFERRED_HEADING = re.compile(
    r"nice[\s-]*to[\s-]*have|preferred|bonus|plus(es)?\b|desirable|good to have", re.I
)
_REQUIRED_HEADING = re.compile(
    r"require|must[\s-]*have|qualification|what you.ll need|minimum", re.I
)
_RESPONSIBILITY_HEADING = re.compile(
    r"responsibilit|what you.ll do|day[\s-]*to[\s-]*day", re.I
)
# Marketing prose about the company is not something a candidate can satisfy.
# Lines under these headings are skipped entirely rather than scored as
# requirements the resume will always "fail".
_CONTEXT_HEADING = re.compile(
    r"about (the role|us|the team|the company)|overview|who we are|our mission", re.I
)

_YEARS = re.compile(r"(\d+)\s*\+?\s*(?:-\s*\d+\s*)?years?", re.I)
_TITLE = re.compile(r"^(?:title|role|position)\s*:\s*(.+)$", re.I)
_COMPANY = re.compile(r"^company\s*:\s*(.+)$", re.I)

_EDUCATION_HINT = re.compile(r"\b(bachelor|master|phd|b\.?s\.?|m\.?s\.?|degree)\b", re.I)


def _kind_for(line: str, in_responsibilities: bool) -> RequirementKind:
    if _EDUCATION_HINT.search(line):
        return RequirementKind.EDUCATION
    if _YEARS.search(line):
        return RequirementKind.EXPERIENCE
    if in_responsibilities:
        return RequirementKind.RESPONSIBILITY
    return RequirementKind.SKILL


def parse_job(text: str) -> JobSpec:
    """Split a posting into requirements, flagged required vs preferred.

    Importance comes from the heading a line sits under, because that is how
    postings actually encode it — a "Nice to have" block means every line in it
    is optional regardless of how it is phrased.
    """
    title = ""
    company = ""
    importance = Importance.REQUIRED
    in_responsibilities = False
    in_context = False
    requirements: list[Requirement] = []

    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue

        match = _TITLE.match(line)
        if match:
            title = match.group(1).strip()
            continue
        match = _COMPANY.match(line)
        if match:
            company = match.group(1).strip()
            continue

        # Headings switch the bucket rather than becoming requirements.
        is_heading = len(line.split()) <= 6 or line.endswith(":")
        if is_heading:
            if _CONTEXT_HEADING.search(line):
                in_context, in_responsibilities = True, False
                continue
            if _PREFERRED_HEADING.search(line):
                importance, in_responsibilities, in_context = (
                    Importance.PREFERRED,
                    False,
                    False,
                )
                continue
            if _REQUIRED_HEADING.search(line):
                importance, in_responsibilities, in_context = (
                    Importance.REQUIRED,
                    False,
                    False,
                )
                continue
            if _RESPONSIBILITY_HEADING.search(line):
                importance, in_responsibilities, in_context = (
                    Importance.REQUIRED,
                    True,
                    False,
                )
                continue

        if in_context:
            continue

        found = skills.extract_skills(line)
        years_match = _YEARS.search(line)
        requirements.append(
            Requirement(
                text=line,
                kind=_kind_for(line, in_responsibilities),
                importance=importance,
                skills=found,
                years=int(years_match.group(1)) if years_match else None,
            )
        )

    if not title:
        # Fall back to the first non-empty line, which is the headline in most
        # postings people paste in.
        for raw in text.splitlines():
            if raw.strip():
                title = raw.strip()
                break

    return JobSpec(
        title=title, company=company, requirements=requirements, raw_text=text
    )
