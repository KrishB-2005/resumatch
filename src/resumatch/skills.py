"""Canonical skill vocabulary, aliases, and extraction from free text.

This is the part that makes "keyword alignment" mean something. A posting says
"JS", the resume says "JavaScript"; a posting says "k8s", the resume says
"Kubernetes". Naive string matching scores both as misses and reports a number
that is simply wrong.

The table below is deliberately hand-written and small. It does not need to
cover every skill in the world — it needs to cover the ones where a literal
comparison gives the wrong answer, and to be auditable when it does. Anything
it misses falls through to the semantic pass, which costs a model call.
"""

from __future__ import annotations

import re

# canonical -> every spelling that should resolve to it.
# Canonical names are lowercase and match how the skill is normally written.
ALIASES: dict[str, tuple[str, ...]] = {
    # languages
    "python": ("py", "python3"),
    "javascript": ("js", "ecmascript", "es6"),
    "typescript": ("ts",),
    "java": (),
    "c++": ("cpp", "cplusplus"),
    "c#": ("csharp",),
    "go": ("golang",),
    "rust": (),
    "ruby": (),
    "sql": ("ansi sql",),
    "r": (),
    "bash": ("shell", "shell scripting"),
    # web / frontend
    "react": ("react.js", "reactjs"),
    "next.js": ("nextjs", "next js"),
    "vue": ("vue.js", "vuejs"),
    "angular": ("angularjs",),
    "tailwind css": ("tailwind", "tailwindcss"),
    "html": ("html5",),
    "css": ("css3",),
    # backend / api
    "node.js": ("node", "nodejs"),
    "flask": (),
    "fastapi": ("fast api",),
    "django": (),
    "express": ("express.js", "expressjs"),
    "rest": ("rest api", "rest apis", "restful", "restful api"),
    "graphql": (),
    "grpc": (),
    "microservices": ("micro services", "microservice"),
    # data / ml
    "pandas": (),
    "numpy": (),
    "scikit-learn": ("sklearn", "scikit learn"),
    "pytorch": ("torch",),
    "tensorflow": ("tf",),
    "machine learning": ("ml",),
    "deep learning": ("dl",),
    "nlp": ("natural language processing",),
    "computer vision": ("cv", "opencv"),
    "data analysis": ("data analytics", "analytics"),
    # llm / genai
    "llm": ("llms", "large language model", "large language models"),
    "rag": ("retrieval augmented generation", "retrieval-augmented generation"),
    "prompt engineering": ("prompting",),
    "langchain": ("lang chain",),
    "langgraph": (),
    "openai": ("openai api", "gpt", "gpt-4", "gpt4"),
    "anthropic": ("claude", "claude api"),
    "vector database": ("vector db", "vector store", "embeddings database"),
    "agents": ("agentic", "agentic workflows", "multi-agent", "multi agent"),
    # cloud / infra
    "aws": ("amazon web services",),
    "gcp": ("google cloud", "google cloud platform"),
    "azure": ("microsoft azure",),
    "docker": ("containerization", "containerisation"),
    "kubernetes": ("k8s",),
    "terraform": ("iac", "infrastructure as code"),
    "ci/cd": ("cicd", "continuous integration", "continuous delivery"),
    "kafka": ("apache kafka",),
    "airflow": ("apache airflow",),
    "serverless": ("lambda", "aws lambda"),
    # storage
    "postgresql": ("postgres", "psql"),
    "mysql": (),
    "mongodb": ("mongo",),
    "redis": (),
    "dynamodb": ("dynamo",),
    "elasticsearch": ("elastic search", "opensearch"),
    "snowflake": (),
    "s3": ("amazon s3",),
    # practice
    "git": ("github", "gitlab", "version control"),
    "testing": ("unit testing", "unit tests", "pytest", "jest", "test automation"),
    "agile": ("scrum", "kanban"),
    "code review": ("code reviews", "peer review"),
    "system design": ("distributed systems", "architecture"),
}

# Reverse index: every spelling (canonical and alias) -> canonical.
_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in ALIASES.items():
    _LOOKUP[_canonical] = _canonical
    for _alias in _aliases:
        _LOOKUP[_alias] = _canonical

# Longest first, so "machine learning" wins over "learning" and "next.js" over
# "next". Scanning shortest-first would fragment multi-word skills.
_PHRASES: list[str] = sorted(_LOOKUP, key=len, reverse=True)

# Skills whose spelling contains regex metacharacters or is short enough that a
# word-boundary match would fire inside unrelated words.
_PUNCT = re.compile(r"[^\w\s+#./-]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation that is never part of a skill, collapse space."""
    lowered = text.lower()
    lowered = _PUNCT.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def _pattern_for(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    # \b does not fire against a leading/trailing non-word char, so "c++" and
    # "c#" need their boundaries asserted by hand.
    left = r"\b" if phrase[0].isalnum() else r"(?<!\w)"
    right = r"\b" if phrase[-1].isalnum() else r"(?!\w)"
    return re.compile(left + escaped + right)


_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (_pattern_for(p), _LOOKUP[p]) for p in _PHRASES
]


def extract_skills(text: str) -> list[str]:
    """Canonical skills mentioned in `text`, in first-appearance order.

    Matched phrases are blanked out as they are consumed so a longer phrase
    cannot be double-counted by a shorter one inside it — "machine learning"
    should not also register the bare "learning" of some other entry.
    """
    haystack = normalise(text)
    found: list[str] = []

    for pattern, canonical in _COMPILED:
        match = pattern.search(haystack)
        if match is None:
            continue
        if canonical not in found:
            found.append(canonical)
        # Blank the span rather than deleting it, so later offsets stay valid.
        haystack = (
            haystack[: match.start()]
            + " " * (match.end() - match.start())
            + haystack[match.end() :]
        )

    return found


def canonical(term: str) -> str | None:
    """Resolve a single term to its canonical name, if it is a known skill."""
    return _LOOKUP.get(normalise(term))


def aliases_of(canonical_name: str) -> tuple[str, ...]:
    return ALIASES.get(canonical_name, ())


def vocabulary_size() -> int:
    return len(ALIASES)
