"""Command line for ResuMatch."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore", message=".*ARC4 has been moved.*")

# Windows consoles default to cp1252, which cannot encode the em dashes and
# arrows this CLI prints; piping output then dies with UnicodeEncodeError
# before anything runs.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

# ruff: noqa: E402

import typer
from dotenv import load_dotenv

from resumatch import benchmark as bench
from resumatch import skills
from resumatch.llm.client import LLMError, build_client
from resumatch.parsing import parse_job, parse_resume, read_text
from resumatch.report import build_report, format_report

app = typer.Typer(
    add_completion=False,
    help="Score a resume against a job description, with every point traceable.",
)

load_dotenv()


@app.command()
def match(
    resume: Annotated[Path, typer.Argument(help="Resume: .pdf, .txt or .md")],
    job: Annotated[Path, typer.Argument(help="Job posting: .pdf, .txt or .md")],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also list what matched.")
    ] = False,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the full report as JSON.")
    ] = None,
    semantic: Annotated[
        bool,
        typer.Option(
            "--semantic",
            help="Ask a model about requirements naming no skill. Costs money.",
        ),
    ] = False,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="openai or fixture. Defaults to openai when OPENAI_API_KEY is set.",
        ),
    ] = None,
    max_calls: Annotated[
        int,
        typer.Option("--max-calls", help="Cap on model calls for --semantic."),
    ] = 12,
) -> None:
    """Match a resume against a posting and print what to change."""
    for path in (resume, job):
        if not path.exists():
            typer.secho(f"No file at {path}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    parsed_resume = parse_resume(read_text(resume))
    parsed_job = parse_job(read_text(job))

    if not parsed_job.requirements:
        typer.secho(
            "No requirements found in the posting. Check it is the full text and "
            "not just a title.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)

    client = None
    if semantic or provider:
        try:
            client = build_client(provider)
        except LLMError as exc:
            typer.secho(f"  {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
        typer.secho(
            f"  semantic pass: {client.name} (up to {max_calls} calls)",
            fg=typer.colors.BLUE,
            err=True,
        )

    report = build_report(
        parsed_job, parsed_resume, llm=client, max_semantic_calls=max_calls
    )
    typer.echo(format_report(report, verbose=verbose))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        typer.secho(f"  wrote {json_out}", fg=typer.colors.GREEN)

    if report.unscoreable:
        typer.echo("  Not scored — no named skill to check against:")
        for item in report.unscoreable:
            typer.echo(f"    ?  {item.requirement.text[:66]}")
            if item.note:
                typer.echo(f"       {item.note[:66]}")
        typer.echo("")


@app.command("read")
def read_command(
    document: Annotated[Path, typer.Argument(help="Resume or posting to inspect.")],
    as_job: Annotated[
        bool, typer.Option("--job", help="Parse as a posting rather than a resume.")
    ] = False,
) -> None:
    """Show how a document was parsed.

    The first thing to check when a score looks wrong.
    """
    if not document.exists():
        typer.secho(f"No file at {document}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    text = read_text(document)
    if as_job:
        parsed = parse_job(text)
        typer.echo(f"\n  {parsed.title}" + (f"  ({parsed.company})" if parsed.company else ""))
        typer.echo(f"  {len(parsed.requirements)} requirements\n")
        for requirement in parsed.requirements:
            mark = "!" if requirement.importance.value == "required" else "."
            found = ", ".join(requirement.skills) or "-"
            typer.echo(
                f"  {mark} [{requirement.kind.value:<14}] "
                f"{found:<34} {requirement.text[:44]}"
            )
    else:
        parsed = parse_resume(text)
        typer.echo(f"\n  {len(parsed.evidence)} lines, {len(parsed.skills())} distinct skills\n")
        for item in parsed.evidence:
            found = ", ".join(item.skills) or "-"
            typer.echo(
                f"  [{item.section.value:<10}] {found:<34} {item.text[:44]}"
            )
    typer.echo("")


@app.command("skills")
def skills_command(
    text: Annotated[
        str | None,
        typer.Argument(help="Text to scan. Omit to list the vocabulary."),
    ] = None,
) -> None:
    """Scan text for known skills, or list the vocabulary."""
    if text:
        found = skills.extract_skills(text)
        typer.echo(json.dumps(found, indent=2) if found else "  no known skills found")
        return

    typer.echo(f"\n  {skills.vocabulary_size()} canonical skills\n")
    for canonical_name, alias_list in sorted(skills.ALIASES.items()):
        shown = f"  {canonical_name}"
        if alias_list:
            shown += f"  ({', '.join(alias_list)})"
        typer.echo(shown)
    typer.echo("")


@app.command("benchmark")
def benchmark_command(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="List every case, not just failures.")
    ] = False,
) -> None:
    """Run the matcher against the hand-labelled cases.

    The number that says whether a change to the vocabulary or the matching
    rules actually helped.
    """
    results = bench.run()
    typer.echo(bench.format_results(results, verbose=verbose))

    failed = [r for r in results if not r.passed]
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
