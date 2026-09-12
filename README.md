# ResuMatch

Scores a resume against a job posting and tells you what to change — with every
point of the score traceable to a specific requirement.

```
$ resumatch match resume.txt job.txt

  Senior AI Platform Engineer
  ---------------------------

  Overall match      53%   (9/17 weighted)
  Required covered   67%

    required   4/6
    preferred  1/5

    matched by  exact=3  alias=2

  Missing, required
    x  Own the serving layer for our retrieval augmented generation stack
    x  Mentor two junior engineers and run code reviews

  What to change
    ! No evidence of code review anywhere on the resume. If you have it, it
      needs a line; if you do not, this is a real gap.
        re: Mentor two junior engineers and run code reviews
    ! Matched on kubernetes only through a synonym. The posting's own wording
      does not appear on your resume — a keyword filter screening on it would
      score this as a miss.
        re: Hands-on with Docker and Kubernetes in production
```

---

## Why not just ask a model

Because "78% match" from a language model is unfalsifiable. You cannot check
it, argue with it, or act on it, and it will give you a different number on
Tuesday.

Every point here is earned weight over total weight, where each unit of weight
belongs to a named requirement with the evidence that satisfied it. If the score
says 53%, you can see exactly which nine of seventeen weighted points landed and
why. That is the property the whole design is arranged around, and it is what
makes the test suite and the benchmark possible.

A model does get used — but only for narrow, checkable questions, and never for
the number itself. See [the semantic pass](#the-semantic-pass).

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

The distinction is finer than it looks, in three ways:

**Same word, different canonical name.** If the posting says *Postgres* and your
resume says *Postgres*, that is a literal hit even though the canonical name is
`postgresql` and appears in neither — so the warning does not fire. It only
fires when the two documents genuinely use different words.

**Per skill, not per requirement.** *"Hands-on with Docker and Kubernetes"* can
be literal on Docker and a synonym on Kubernetes at the same time. The first
implementation marked the whole requirement exact as soon as anything in it
matched literally, which swallowed the warning in exactly the case it was
written for.

**Whole words, not substrings.** `go` is inside `golang` and `node` is inside
`node.js`. A substring test reports that both documents used the same word when
one of them said something else entirely — and suppresses the warning again.

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

An honest 53% over what was actually checkable beats a confident 39% that
counted unanswerable questions as failures.

## How it works

```
resume ─┐
        ├─→ parse ─→ canonical skills ─→ match ─→ score ─→ advice
job   ──┘                                  ↑
                                    semantic pass (optional)
```

**Parsing** is heuristic and deterministic. Resume sections come from heading
detection; requirement importance comes from the heading a line sits under,
because that is how postings actually encode it — everything under *Nice to
have* is optional regardless of how it is phrased. Company marketing prose under
*About us* is skipped rather than scored, and the block ends at the next heading
whether or not that heading is one the parser recognises.

Three rules here exist because documents are messier than they look. A bullet is
never a heading, so *"• Strong Python required"* stays a requirement instead of
reading as a *Requirements* label. A wrapped line rejoins the one above it, so a
bullet spanning two lines is one requirement rather than two, the second a
fragment. And everything above the first recognised heading is masthead — job
title, company, location — rather than requirements no candidate could ever
meet. All three are measured by the parsing benchmark below.

**Canonicalisation** runs a hand-written alias table over 166 skills, longest
phrase first so `machine learning` wins over the bare `learning` inside it, and
matched spans are blanked as they are consumed so nothing is double-counted.
Word boundaries are asserted by hand for `c++`, `c#` and `node.js`, where `\b`
does not fire. Plurals are matched against the singular entry, so *"exposure to
vector databases"* is not silently invisible.

**Ambiguity** is handled by denying idioms rather than words. `go`, `rest`,
`react`, `express`, `spark`, `swift`, `rails` and `helm` are all ordinary
English, and a phantom skill is worse than a missed one — it invents evidence
the candidate never claimed. So *"go to production"*, *"the rest of the team"*
and *"went off the rails"* are prose, while *"experience with Go"* and *"built
the dashboard in React"* are skills. The denial lists are short, hand-written,
and sit in one table where a wrong entry is visible instead of buried in a score.

**Matching** tries literal agreement first, then the alias table, per skill. It
also reads *and* versus *or*: *"Docker and Kubernetes"* is not satisfied by
Docker alone, while *"React or TypeScript"* is satisfied by either. Treating
every multi-skill requirement as a choice gave resumes full marks on
requirements they half met.

**Advice** is rule-based, not generated. Each rule fires off a fact the matcher
already established, so every suggestion names the requirement it closes and the
specific skill to change.

## The semantic pass

Optional, off by default, and the only part that calls a model.

It runs last, on the requirements literal and alias matching could not reach —
prose responsibilities like *"partner with research to take prototypes to
production"*, which name no skill and so have nothing to check.

```bash
resumatch match resume.txt job.txt --semantic
```

Four constraints keep it from becoming the thing it was built to avoid.

**It is shown a shortlist, not the resume.** Candidate lines are ranked with
BM25 first and the top six are sent. A model handed forty lines and asked "does
any of this count?" drifts toward yes.

**It answers one question at a time.** It never sees the score, the other
requirements, or the candidate's name — only "do these specific lines satisfy
this specific requirement?".

**A yes must cite a line.** A satisfied verdict with no citation is refused and
recorded as unmatched, because that is precisely the unfalsifiable answer the
design exists to reject. So is a verdict below 0.6 confidence.

**It can lower the score.** A verdict either way makes the requirement
scoreable, so a *no* moves it from "not scored" into the denominator as a miss.
Scoring only the yes verdicts would make every run an improvement, which is a
score-inflation device dressed up as a judgement.

Semantic matches are reported as `semantic`, never merged into `exact` — "this
was literally on the resume" and "a model thought this counted" are different
claims and should not look the same.

### Providers

| | |
|---|---|
| `openai` | `responses.parse(..., text_format=...)`, default `gpt-4o` |
| `anthropic` | `messages.parse(..., output_format=...)`, default `claude-sonnet-5` |
| `fixture` | no network, no key, no spend — **the default** |

Whichever key is in the environment decides the provider; with neither, it falls
back to `fixture`, which declines every judgement. A stub that answered "yes"
would make the keyless path outscore the real one and quietly inflate every
number in the test suite.

```bash
resumatch match resume.txt job.txt --semantic --provider anthropic --max-calls 8
```

Both request shapes are tested against the real SDKs pointed at a loopback
server — no key, no network, but the request is built by the same code that
would talk to the live endpoint and the response is decoded by the same parser.
That is what caught `confidence: Field(ge=0, le=1)`: Pydantic emits it as
`minimum`/`maximum`, both providers reject numeric range keywords in strict
mode, and every semantic call would have failed with a 400 on a real key.

## Benchmarks

"I improved the matcher" is not a claim anyone can check.

```bash
resumatch benchmark
```

```
  matching

  30/30 cases

  no failures

  parsing

  8/8 documents

  no failures
```

Two suites, because they fail in different places.

**matching** is 30 hand-labelled requirement/resume pairs. It measures the
scoring rules starting from clean sentences.

**parsing** is 8 whole postings and resumes in the shapes people actually send
them — a headline instead of a `Title:` field, numbered lists, all-caps
headings, requirements written as paragraphs, one technology per line with no
bullet, and a resume mangled the way `pypdf` mangles a two-column PDF. It
measures whether the right sentences came out at all.

That second suite is the one that matters most, and it did not exist at first.
Every test above it passed while the parser was quietly wrong, because they all
fed it sentences somebody had already extracted by hand. Run whole documents
through and it scored **2/7** on the first attempt:

- A posting whose next heading after *"Our mission"* was *"The role"* parsed to
  a single requirement — its own title. The context block never closed, so the
  candidate was scored against nothing.
- `• Strong Python required` is short and contains "required", so it read as a
  *Requirements* heading and vanished from the posting.
- Wrapped bullets became two requirements, the second a fragment — *"and
  backwards compatibility across client versions"* — naming no skill and
  impossible to satisfy.
- The job title and company line were scored as requirements: permanent misses
  no candidate can fix.
- A phone number became resume evidence, and `github.com/someone` handed the
  candidate a `git` skill they never claimed.
- *"Five or more years"* extracted no tenure at all, because the pattern only
  matched digits.

Both suites carry, per case, the reasoning a human would give, so a failure can
be argued with — sometimes the right fix is the label. Failing cases stay in the
file rather than being deleted, and tests assert a floor on the case count and
that the interesting categories are still represented, because pruning is the
easiest way to make a score go up.

## Install and run

```bash
git clone https://github.com/KrishB-2005/resumatch
cd resumatch
pip install -e ".[dev]"

resumatch match data/samples/resume.txt data/samples/job.txt
```

Takes `.pdf`, `.txt` or `.md` for either document. For the semantic pass:

```bash
pip install -e ".[dev,llm]"
cp .env.example .env    # then set one key
```

### Commands

| | |
|---|---|
| `match` | score a resume against a posting and print what to change |
| `read` | show how a document was parsed — check this first when a score looks wrong |
| `skills` | scan text for known skills, or list the vocabulary |
| `benchmark` | run the hand-labelled cases — `--suite matching`, `--suite parsing`, or both |

```bash
resumatch match resume.pdf job.txt --verbose --json out/report.json
```

```bash
resumatch read job.txt --job
```

```bash
resumatch skills "Built REST APIs in Python and JS, deployed on k8s"
# ["rest", "python", "kubernetes", "javascript"]
```

## Tests

No API key, no network, no spend. The matcher is not allowed to need a model,
and the tests hold it to that — the semantic pass is exercised with a scripted
provider that can be made to return exactly the answer worth testing.

```bash
pytest
```

140 tests. Most of them exist because the first implementation got something
wrong:

- Comparing the whole requirement sentence against a resume line is never true,
  so every match reported as an alias hit and the keyword warning fired on
  everything. Now it compares per skill, and only when the two documents use
  different spellings.
- Requirements naming no skill were scored as misses, reporting 39% where the
  honest number over checkable requirements was 56%.
- A literal hit on one skill hid a synonym on another in the same requirement,
  suppressing the warning in the exact case the README leads with.
- `"go to production"` scored as the Go language, and `"the rest of the team"`
  as REST — phantom skills that let a resume match requirements it never spoke
  to.
- Every multi-skill requirement was treated as a choice, so a resume with
  Docker alone scored full marks against "Docker and Kubernetes".
- `confidence` carried a range constraint that both model providers reject.
- Six separate parser bugs, all listed above, none of which any unit test could
  see because none of them fed it a whole document.

## Status

Deterministic core and semantic pass are done, tested and benchmarked at both
layers. Still to build:

- **Rewrite suggestions** — given a gap and the resume's voice, draft the line.
- **Harder benchmark cases.** Both suites currently pass completely, which
  means neither has headroom left to measure anything. Negations ("no
  Kubernetes required"), seniority mismatches, and skills outside the
  vocabulary are the obvious next cases.
- **A real model behind the semantic pass.** It has only ever talked to a
  scripted stub and a loopback server, so whether a live model actually honours
  the cite-your-evidence rule is untested.
- **Web UI**, the same way [TaxOrchestra](https://github.com/KrishB-2005/taxorchestra)
  does it: everything client-side, because a resume is a personal document and
  should not need a server.
