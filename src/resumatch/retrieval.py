"""Pick the resume lines worth asking a model about.

The semantic pass exists to judge requirements that literal and alias matching
could not reach. Handing it the whole resume for every such requirement is both
expensive and worse: a model given forty lines and asked "does any of this
count?" drifts toward yes.

So candidates are ranked lexically first and the top handful are sent. The
ranking is BM25-style over the resume's own lines, which needs no model, no
embeddings and no index build, and returns the same order every time — which is
what lets the semantic pass be tested with a stubbed provider.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from resumatch.models import Evidence, Requirement

_TOKEN = re.compile(r"[a-z0-9+#.]+")

# Words that appear in nearly every posting and nearly every resume. Leaving
# them in lets "experience with" match "experience in" and rank noise highly.
_STOPWORDS = frozenset(
    """
    a an and are as at be been build building built by can develop developing
    for from has have in including into is it its of on or our strong that the
    their them they this to us use used using with within work working you your
    experience experienced knowledge familiar familiarity ability able skills
    """.split()
)

K1 = 1.5
B = 0.75

# How many lines the semantic pass is allowed to see per requirement. Enough to
# find the answer, few enough that a model cannot pad its way to a yes.
DEFAULT_TOP_K = 6

# Below this, the line has essentially nothing in common with the requirement
# and sending it only adds noise.
MIN_SCORE = 0.5


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Candidate:
    evidence: Evidence
    score: float


class EvidenceIndex:
    """BM25 over the resume's evidence lines."""

    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        self._docs = [tokenize(item.text) for item in evidence]
        self._lengths = [len(d) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._docs)) if self._docs else 0.0
        self._tf = [Counter(d) for d in self._docs]

        frequency: Counter[str] = Counter()
        for doc in self._docs:
            frequency.update(set(doc))

        total = len(self._docs)
        self._idf = {
            term: max(math.log((total - count + 0.5) / (count + 0.5) + 1.0), 0.0)
            for term, count in frequency.items()
        }

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Candidate]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []

        scored: list[Candidate] = []
        for index, item in enumerate(self.evidence):
            length = self._lengths[index] or 1
            counts = self._tf[index]
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = freq + K1 * (1 - B + B * length / (self._avg_len or 1))
                score += idf * (freq * (K1 + 1)) / denominator
            if score >= MIN_SCORE:
                scored.append(Candidate(evidence=item, score=score))

        # Ties broken by resume order so the ranking is fully deterministic.
        scored.sort(key=lambda c: (-c.score, self.evidence.index(c.evidence)))
        return scored[:top_k]

    def candidates_for(
        self, requirement: Requirement, top_k: int = DEFAULT_TOP_K
    ) -> list[Evidence]:
        return [c.evidence for c in self.search(requirement.text, top_k)]
