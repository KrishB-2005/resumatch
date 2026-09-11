"""Model access, behind one narrow interface.

Three providers:

  openai     `responses.parse(..., text_format=Model)`
  anthropic  `messages.parse(..., output_format=Model)`
  fixture    No network, no key, no spend. Returns a conservative verdict so
             the pipeline, the tests and the CLI all work without either SDK.

`fixture` is the default. Nothing here reaches the network unless a key is
already in the environment and the user asked for the semantic pass, so the
tool cannot quietly start costing money.

The interface is deliberately one method returning a Pydantic instance. Every
question this project asks a model is narrow and checkable — "do these specific
lines satisfy this specific requirement?" — never "how good is this candidate?",
which is the question that produces unfalsifiable numbers.

One constraint the two SDKs share, worth knowing before editing a schema: both
enforce strict JSON-schema output, and neither accepts numeric range keywords.
`Field(ge=0, le=1)` becomes `minimum`/`maximum` and is rejected with a 400, so
bounds belong in a validator rather than the field. `test_wire_format.py` holds
both providers to that.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Overridable with RESUMATCH_MODEL. Both default to a mid-tier model: the
# question asked here is small and well-specified, and the frontier models cost
# several times more to answer it no better.
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# Every call here returns a small JSON object. This is headroom, not a target.
MAX_OUTPUT_TOKENS = 2048


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str
    calls: int

    def parse(self, *, instructions: str, user: str, schema: type[T]) -> T: ...


def _require_key(variable: str, provider: str) -> None:
    if not os.getenv(variable):
        raise LLMError(
            f"{variable} is not set, so provider {provider!r} cannot run. "
            f"Export it, put it in .env, or use --provider fixture to stay "
            f"offline and free."
        )


class OpenAIClient:
    """OpenAI, via the Responses API's structured-output path."""

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        try:
            import openai
        except ModuleNotFoundError as exc:  # pragma: no cover - install-time
            raise LLMError(
                "provider 'openai' needs the SDK: pip install 'resumatch[llm]'"
            ) from exc

        _require_key("OPENAI_API_KEY", "openai")

        self._openai = openai
        self._client = openai.OpenAI()
        self.model = model or os.getenv("RESUMATCH_MODEL", DEFAULT_OPENAI_MODEL)
        self.calls = 0

    def parse(self, *, instructions: str, user: str, schema: type[T]) -> T:
        self.calls += 1
        try:
            response = self._client.responses.parse(
                model=self.model,
                instructions=instructions,
                input=user,
                text_format=schema,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        except self._openai.AuthenticationError as exc:
            raise LLMError("authentication failed — check OPENAI_API_KEY") from exc
        except self._openai.NotFoundError as exc:
            raise LLMError(
                f"model {self.model!r} not available on this key — set RESUMATCH_MODEL"
            ) from exc
        except self._openai.BadRequestError as exc:
            raise LLMError(f"malformed request: {exc}") from exc
        except self._openai.RateLimitError as exc:
            raise LLMError("rate limited; the SDK already retried") from exc
        except self._openai.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except self._openai.APIConnectionError as exc:
            raise LLMError(f"network error reaching the OpenAI API: {exc}") from exc

        parsed = response.output_parsed
        if parsed is None:
            # Structured output comes back empty if the model stopped early —
            # hitting the token ceiling, or a refusal.
            raise LLMError(
                f"model returned no parseable output (status={response.status})"
            )
        return parsed


class AnthropicClient:
    """Claude, via the Messages API's structured-output path."""

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        try:
            import anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - install-time
            raise LLMError(
                "provider 'anthropic' needs the SDK: pip install 'resumatch[llm]'"
            ) from exc

        _require_key("ANTHROPIC_API_KEY", "anthropic")

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model or os.getenv("RESUMATCH_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.calls = 0

    def parse(self, *, instructions: str, user: str, schema: type[T]) -> T:
        self.calls += 1
        try:
            message = self._client.messages.parse(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=instructions,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except self._anthropic.AuthenticationError as exc:
            raise LLMError("authentication failed — check ANTHROPIC_API_KEY") from exc
        except self._anthropic.NotFoundError as exc:
            raise LLMError(
                f"model {self.model!r} not available on this key — set RESUMATCH_MODEL"
            ) from exc
        except self._anthropic.BadRequestError as exc:
            raise LLMError(f"malformed request: {exc}") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError("rate limited; the SDK already retried") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"network error reaching the Anthropic API: {exc}") from exc

        parsed = message.parsed_output
        if parsed is None:
            raise LLMError(
                f"model returned no parseable output "
                f"(stop_reason={message.stop_reason})"
            )
        return parsed


class FixtureClient:
    """Deterministic stand-in. Declines rather than guesses.

    A stub that returned "yes, satisfied" would make the keyless path look
    better than the real one and would quietly inflate every score in the test
    suite. Declining keeps the deterministic result honest: unreachable
    requirements stay unreachable until a real provider is configured.
    """

    name = "fixture"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "fixture-deterministic"
        self.calls = 0

    def parse(self, *, instructions: str, user: str, schema: type[T]) -> T:
        self.calls += 1
        payload: dict[str, object] = {}
        for name, field in schema.model_fields.items():
            if name in ("satisfied", "matched"):
                payload[name] = False
            elif name == "confidence":
                payload[name] = 0.0
            elif name in ("evidence_indices", "indices"):
                payload[name] = []
            elif field.annotation is str:
                payload[name] = "fixture provider: no semantic judgement made"
            elif field.is_required():
                payload[name] = "" if field.annotation is str else None
        return schema.model_validate(payload)


_PROVIDERS: dict[str, type] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "claude": AnthropicClient,
    "fixture": FixtureClient,
}

# Which key implies which provider, in the order they are tried.
_KEYS = (("OPENAI_API_KEY", "openai"), ("ANTHROPIC_API_KEY", "anthropic"))


def build_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Resolve a provider by name, falling back to fixture without a key.

    Falling back rather than failing is deliberate: the tool's whole value is
    the deterministic match, and that half works with no key at all.
    """
    name = (provider or os.getenv("RESUMATCH_PROVIDER", "")).strip().lower()

    if not name:
        # Pick what will actually work rather than failing on a missing key.
        name = next(
            (p for variable, p in _KEYS if os.getenv(variable)),
            "fixture",
        )

    try:
        factory = _PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise LLMError(f"unknown provider {name!r}; expected one of {known}") from None
    return factory(model)
