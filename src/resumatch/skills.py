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
    "rest": ("rest api", "restful", "restful api"),
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
    "testing": ("unit testing", "unit test", "pytest", "jest", "test automation"),
    "agile": ("scrum", "kanban"),
    "code review": ("peer review",),
    "system design": ("distributed systems", "architecture"),
    # more languages
    "scala": (),
    "kotlin": (),
    "swift": ("swiftui",),
    "php": (),
    "matlab": (),
    "julia": (),
    "elixir": (),
    "solidity": (),
    # more web
    "svelte": ("sveltekit",),
    "redux": (),
    "webpack": (),
    "vite": (),
    "sass": ("scss",),
    "bootstrap": (),
    "jquery": (),
    "figma": (),
    "accessibility": ("a11y", "wcag"),
    "responsive design": ("mobile first",),
    # more backend
    "spring": ("spring boot",),
    "rails": ("ruby on rails",),
    ".net": ("dotnet", "asp.net"),
    "celery": (),
    "rabbitmq": ("rabbit mq",),
    "websockets": ("websocket", "socket.io"),
    "oauth": ("oauth2", "openid connect", "sso"),
    "jwt": ("json web token",),
    "api design": ("api development",),
    "caching": ("cache", "memcached"),
    # more data
    "spark": ("apache spark", "pyspark"),
    "hadoop": ("hdfs", "mapreduce"),
    "dbt": (),
    "databricks": (),
    "tableau": (),
    "power bi": ("powerbi",),
    "looker": (),
    "bigquery": ("big query",),
    "redshift": (),
    "etl": ("elt", "data pipeline", "data pipelines"),
    "data modeling": ("data modelling", "dimensional modeling", "star schema"),
    "sqlalchemy": (),
    "matplotlib": (),
    "jupyter": ("jupyter notebook", "notebooks"),
    "statistics": ("statistical analysis", "statistical modeling"),
    "a/b testing": ("ab testing", "split testing", "experimentation"),
    # more ml
    "xgboost": ("gradient boosting", "lightgbm"),
    "keras": (),
    "hugging face": ("huggingface", "transformers library"),
    "mlops": ("model deployment", "model serving"),
    "mlflow": (),
    "cuda": (),
    "feature engineering": (),
    "time series": ("forecasting", "time-series"),
    "recommendation systems": ("recommender systems", "recsys"),
    "reinforcement learning": ("rl",),
    # more llm
    "llamaindex": ("llama index",),
    "pinecone": (),
    "weaviate": (),
    "fine-tuning": ("fine tuning", "finetuning", "lora", "peft"),
    "embeddings": ("sentence embeddings", "text embeddings"),
    "semantic search": ("similarity search",),
    "evals": ("evaluation harness", "model evaluation"),
    "mcp": ("model context protocol",),
    # more cloud / infra
    "linux": ("unix",),
    "nginx": (),
    "ansible": (),
    "helm": (),
    "jenkins": (),
    "github actions": ("gh actions",),
    "prometheus": (),
    "grafana": (),
    "datadog": ("data dog",),
    "observability": ("opentelemetry", "distributed tracing"),
    "monitoring": ("alerting", "on-call", "oncall", "incident response"),
    "load balancing": ("load balancer",),
    "cdn": ("cloudfront", "content delivery network"),
    # more storage
    "cassandra": (),
    "neo4j": ("graph database",),
    "sqlite": (),
    "clickhouse": ("click house",),
    "supabase": (),
    "firebase": (),
    "prisma": (),
    # more practice
    "tdd": ("test driven development", "test-driven development"),
    "oop": ("object oriented", "object-oriented programming"),
    "functional programming": (),
    "concurrency": ("multithreading", "async programming", "parallelism"),
    "performance optimization": ("performance tuning", "profiling", "optimisation"),
    "refactoring": ("technical debt",),
    "debugging": ("root cause analysis", "troubleshooting"),
    "technical writing": ("documentation", "design docs", "rfcs"),
    "mentoring": ("coaching", "onboarding"),
    "security": ("appsec", "application security", "threat modeling", "pentesting"),
    "stakeholder management": ("cross-functional", "cross functional"),
    "product management": ("product sense", "roadmapping"),
    # "linear" is a real tracker, but "linear growth" and "linear regression"
    # are far more common in the documents this reads. Not worth the noise.
    "jira": ("confluence",),
    "design patterns": (),
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


# Below this length a trailing "s" is more often a different word than a
# plural — "go"/"goes" and "r"/"rs" are the ones that actually bite.
_MIN_PLURAL_LENGTH = 4


def _pattern_for(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    # \b does not fire against a leading/trailing non-word char, so "c++" and
    # "c#" need their boundaries asserted by hand.
    left = r"\b" if phrase[0].isalnum() else r"(?<!\w)"

    if not phrase[-1].isalpha():
        return re.compile(left + escaped + r"(?!\w)")

    # Postings say "vector databases" and "microservices" while the vocabulary
    # holds the singular. Without this the plural is simply not seen, because
    # \b will not match between "database" and the "s" that follows it.
    plural = r"(?:e?s)?" if len(phrase) >= _MIN_PLURAL_LENGTH else ""
    return re.compile(left + escaped + plural + r"\b")


_COMPILED: list[tuple[re.Pattern[str], str, str]] = [
    (_pattern_for(p), _LOOKUP[p], p) for p in _PHRASES
]

# Canonical names that are also ordinary English words. Spelled bare, "go to
# production", "the rest of the team" and "R&D" all look like skills to a
# word-boundary match, and a phantom skill is worse than a missed one — it
# invents evidence the candidate never claimed.
#
# The discriminator is the next word, not the surrounding context. "go to
# production" and "the rest of the team" give themselves away immediately,
# where "experience with Go" and "Built the dashboard in React" do not — and a
# rule that demanded corroborating skills would throw those two away, since
# neither line mentions anything else.
#
# Hand-written and short on purpose: each entry is one idiom that actually
# turns up in resumes, and an entry that is wrong is visible here rather than
# buried in a score. Distinctive aliases ("golang", "rest api", "express.js")
# are never ambiguous and skip the check entirely.
_VERB_SENSE: dict[str, frozenset[str]] = {
    "go": frozenset(
        {
            "to", "live", "ahead", "back", "forward", "home", "through",
            "into", "over", "beyond", "around", "down", "up", "out", "away",
            "past", "straight", "wrong", "well", "deep", "public",
        }
    ),
    "rest": frozenset({"of", "day", "days", "assured"}),
    # From "R&D", which normalises to "r d".
    "r": frozenset({"d"}),
    "react": frozenset({"to"}),
    "express": frozenset({"the", "a", "an", "my", "his", "her", "their", "our"}),
    "spark": frozenset(
        {"innovation", "interest", "joy", "creativity", "curiosity",
         "conversation", "conversations", "debate", "ideas", "change"}
    ),
    # Adjective rather than verb, but it reads the same way: "a swift
    # resolution" is not the language.
    "swift": frozenset(
        {"resolution", "turnaround", "response", "action", "decision",
         "delivery", "execution", "fix", "feedback", "progress", "adoption"}
    ),
    "helm": frozenset({"of"}),
}

# The same idea looking backwards, as whole phrases rather than single words —
# the tell in "went off the rails" is two words back, not one.
_IDIOM_BEFORE: dict[str, tuple[str, ...]] = {
    "rails": ("off the",),
}

_NEXT_WORD = re.compile(r"\s*([a-z]+)")


def _is_verb_sense(
    canonical_name: str, phrase: str, haystack: str, span: tuple[int, int]
) -> bool:
    """Whether the words around the match give it away as ordinary English."""
    following = _VERB_SENSE.get(canonical_name)
    if following:
        nxt = _NEXT_WORD.match(haystack, span[1])
        if nxt is not None and nxt.group(1) in following:
            return True

    before = haystack[: span[0]].rstrip()
    return any(before.endswith(idiom) for idiom in _IDIOM_BEFORE.get(phrase, ()))


def extract_skills(text: str) -> list[str]:
    """Canonical skills mentioned in `text`, in first-appearance order.

    Matched phrases are blanked out as they are consumed so a longer phrase
    cannot be double-counted by a shorter one inside it — "machine learning"
    should not also register the bare "learning" of some other entry.

    Names that are also ordinary English words get one extra check first — see
    _VERB_SENSE — so that prose containing the word "go" does not become a
    claim about the Go language.
    """
    haystack = normalise(text)
    found: list[str] = []

    for pattern, canonical, phrase in _COMPILED:
        bare = phrase == canonical
        position = 0

        while (match := pattern.search(haystack, position)) is not None:
            span = match.span()
            # Blank the span rather than deleting it, so later offsets stay
            # valid. Denied spans are blanked too: the word is still spoken
            # for, and leaving it would only let a shorter pattern find it.
            haystack = (
                haystack[: span[0]] + " " * (span[1] - span[0]) + haystack[span[1] :]
            )
            position = span[1]

            # Only a bare mention is ambiguous — "golang" and "rest api" are
            # not. A denied mention does not end the scan: "react to incidents
            # in our React app" says both, and the second one counts.
            if bare and _is_verb_sense(canonical, phrase, haystack, span):
                continue

            if canonical not in found:
                found.append(canonical)
            break

    return found


def mentions(text: str, phrase: str) -> bool:
    """Whether `text` uses this exact spelling, as a whole word.

    Substring containment is not good enough here: "go" is inside "golang" and
    "node" is inside "node.js", so a substring test reports that both documents
    used the same word when one of them said something else entirely.
    """
    return _pattern_for(normalise(phrase)).search(normalise(text)) is not None


def spelling_used(text: str, canonical_name: str) -> str | None:
    """Which of a skill's spellings `text` actually uses, longest first.

    Longest wins so "node.js" is not reported as the bare "node" that sits
    inside it — the distinction is the whole point of the alias warning.
    """
    for phrase in sorted(
        (canonical_name, *aliases_of(canonical_name)), key=len, reverse=True
    ):
        if mentions(text, phrase):
            return phrase
    return None


def canonical(term: str) -> str | None:
    """Resolve a single term to its canonical name, if it is a known skill."""
    return _LOOKUP.get(normalise(term))


def aliases_of(canonical_name: str) -> tuple[str, ...]:
    return ALIASES.get(canonical_name, ())


def vocabulary_size() -> int:
    return len(ALIASES)
