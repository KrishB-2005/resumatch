# ResuMatch

Scores a resume against a job posting and tells you what to change — with every
point of the score traceable to a specific requirement.

```
$ resumatch match resume.txt job.txt

  Senior AI Platform Engineer
  ---------------------------

  Overall match      56%   (9/16 weighted)
  Required covered   67%

    required   4/6
    preferred  1/4

  Missing, required
    x  Own the serving layer for our retrieval augmented generation stack
    x  Mentor two junior engineers and run code reviews

  Missing, nice to have
    -  Familiarity with LangChain or similar orchestration frameworks
    -  Terraform or other infrastructure-as-code tooling

  What to change
    ! No evidence of rag anywhere on the resume. If you have it, it needs a
      line; if you do not, this is a real gap.
        re: Own the serving layer for our retrieval augmented generation stack
    . Supported only by unquantified lines, e.g. "Shipped an internal dashboard
      in React and TS". A number here would carry more weight.
```

---

## Why not just ask a model

Because "78% match" from a language model is unfalsifiable. You cannot check
it, argue with it, or act on it, and it will give you a different number on
Tuesday.

Every point here is earned weight over total weight, where each unit of weight
belongs to a named requirement with the evidence that satisfied it. If the score
says 56%, you can see exactly which nine of sixteen weighted points landed and
why. That is the property the whole design is arranged around, and it is what
makes the test suite possible.

The model is not absent from the plan — it is just not allowed to be the thing
producing the number.

## The bit that actually matters

A posting says **k8s**. Your resume says **Kubernetes**. A human reading it makes
the connection instantly; a keyword filter sitting in front of that human does
not, and you never hear back.

ResuMatch resolves both to a canonical skill so the *match* is right, then tells
you the wording diverged:

```
! Matched on kubernetes only through a synonym. The posting's own wording does
  not appear on your resume — a keyword filter screening on it would score this
  as a miss.
    re: Hands-on with Docker and Kubernetes in production
```

That is the single highest-value line in the report, and it is the thing a
generic "tailor your resume to the job" suggestion never gives you.

The distinction is finer than it looks. If the posting says *Postgres* and your
resume says *Postgres*, that is a literal hit even though the canonical name is
`postgresql` and appears in neither — so the warning does not fire. It only
fires when the two documents genuinely use different words.

## What it does not score

A requirement that names no skill — *"4+ years of professional backend
experience"*, *"partner with research to take prototypes to production"* — has
nothing for a deterministic matcher to check. Counting those as misses would
deflate the score with things the tool never assessed, so they are reported
separately for a human to check:

```
  Not scored — no named skill to check against:
    ?  4+ years of professional backend engineering experience
    ?  Bachelor's degree in Computer Science or equivalent experience
```

An honest 56% over what was actually checkable beats a confident 39% that
counted unanswerable questions as failures.

## How it works

```
resume ─┐
        ├─→ parse ─→ canonical skills ─→ match ─→ score ─→ advice
job   ──┘
```

**Parsing** is heuristic and deterministic. Resume sections come from heading
detection; requirement importance comes from the heading a line sits under,
because that is how postings actually encode it — everything under *Nice to
have* is optional regardless of how it is phrased. Company marketing prose under
*About us* is skipped rather than scored.

**Canonicalisation** runs a hand-written alias table, longest phrase first so
`machine learning` wins over the bare `learning` inside it, and matched spans
are blanked as they are consumed so nothing is double-counted. Word boundaries
are asserted by hand for `c++`, `c#` and `node.js`, where `\b` does not fire.

**Matching** tries literal agreement first, then the alias table. The method is
recorded on every match, which is what lets the advice distinguish "you don't
have this" from "you have this but called it something else".

**Advice** is rule-based, not generated. Each rule fires off a fact the matcher
already established, so every suggestion names the requirement it closes.

## Install and run

```bash
git clone https://github.com/KrishB-2005/resumatch
cd resumatch
pip install -e ".[dev]"

resumatch match data/samples/resume.txt data/samples/job.txt
```

Takes `.pdf`, `.txt` or `.md` for either document.

### Commands

| | |
|---|---|
| `match` | score a resume against a posting and print what to change |
| `read` | show how a document was parsed — check this first when a score looks wrong |
| `skills` | scan text for known skills, or list the vocabulary |

```bash
resumatch match resume.pdf job.txt --verbose --json out/report.json
```

```bash
resumatch read job.txt --job
```

```bash
resumatch skills "Built REST APIs in Python and JS, deployed on k8s"
# ["rest", "python", "docker", "kubernetes", "javascript"]
```

## Tests

No API key, no network, no spend — the matcher is not allowed to need a model,
and the tests hold it to that.

```bash
pytest
```

Two of them exist because the first implementation got them wrong:

- Comparing the whole requirement sentence against a resume line is never true,
  so every match reported as an alias hit and the keyword warning fired on
  everything. Now it compares per skill, and only when the two documents use
  different spellings.
- Requirements naming no skill were scored as misses, reporting 39% where the
  honest number over checkable requirements was 56%.

## Status

The deterministic core is done and tested. Still to build:

- **Semantic pass** — one model call for the requirements literal and alias
  matching could not reach, so prose responsibilities stop being unscoreable.
  `MatchMethod.SEMANTIC` is already in the model for it.
- **Rewrite suggestions** — given a gap and the resume's voice, draft the line.
- **Web UI**, the same way [TaxOrchestra](https://github.com/KrishB-2005/taxorchestra)
  does it: everything client-side, because a resume is a personal document and
  should not need a server.

MIT.
