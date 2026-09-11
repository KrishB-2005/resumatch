"""What the provider clients actually put on the wire.

The scripted client in `test_semantic.py` proves the *logic* is right. It cannot
catch the class of bug that only shows up against a real API: a request the
service rejects, or a response the SDK cannot decode. Both fail at runtime, on
someone else's key, after the tool has already told them it is working.

So these tests point the real SDKs at a loopback server that records the request
and replies with a canned payload. No key, no network, no spend — but the
request is built by the same SDK code that would talk to the real endpoint, and
the response is decoded by the same parser.

The bug this caught on the way in: `confidence` was declared `Field(ge=0, le=1)`,
which Pydantic emits as `minimum`/`maximum`. Both providers reject numeric range
keywords in strict mode, so every semantic call would have failed with a 400.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from resumatch.agents import INSTRUCTIONS, SemanticVerdict

# Keywords neither provider accepts inside a strict output schema. Pydantic
# emits all of these from ordinary `Field(...)` constraints, which is what makes
# this worth asserting rather than remembering.
UNSUPPORTED_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
)

VERDICT_JSON = json.dumps(
    {
        "satisfied": True,
        "evidence_indices": [1],
        "confidence": 0.82,
        "reasoning": "line 1 names weekly code reviews for junior engineers",
    }
)

PROMPT = (
    "Requirement (required): Mentor junior engineers\n\n"
    "Candidate resume lines:\n1. [experience] Ran weekly code reviews"
)


class Loopback:
    """A one-route HTTP server that records the request and replies canned."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.request: dict[str, Any] = {}
        body = json.dumps(response).encode()
        recorder = self.request

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                recorder["path"] = self.path
                recorder["headers"] = dict(self.headers)
                recorder["body"] = json.loads(self.rfile.read(length) or b"{}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> Loopback:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def openai_response() -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": VERDICT_JSON, "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": VERDICT_JSON, "citations": None}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40},
    }


# -- the schema both providers have to accept ------------------------------


def test_the_verdict_schema_uses_no_keyword_strict_mode_rejects() -> None:
    """A bound in the schema is a 400. A bound in a validator is free."""
    schema = json.dumps(SemanticVerdict.model_json_schema())

    offenders = [k for k in UNSUPPORTED_KEYWORDS if f'"{k}"' in schema]
    assert offenders == [], (
        f"{offenders} would be rejected by strict structured output — move the "
        f"constraint into a field_validator"
    )


def test_the_bound_is_still_enforced_after_decoding() -> None:
    """Dropping ge/le from the schema must not drop the guarantee."""
    assert SemanticVerdict(satisfied=True, confidence=1.7, reasoning="x").confidence == 1.0
    assert SemanticVerdict(satisfied=True, confidence=-3.0, reasoning="x").confidence == 0.0


# -- openai ----------------------------------------------------------------


def test_openai_sends_the_request_the_responses_api_expects(monkeypatch) -> None:
    openai = pytest.importorskip("openai")
    assert openai  # the SDK builds the request; this test only inspects it

    from resumatch.llm.client import OpenAIClient

    with Loopback(openai_response()) as server:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
        monkeypatch.setenv("OPENAI_BASE_URL", f"{server.url}/v1")

        result = OpenAIClient().parse(
            instructions=INSTRUCTIONS, user=PROMPT, schema=SemanticVerdict
        )

    body = server.request["body"]
    assert server.request["path"] == "/v1/responses"
    assert body["model"] == "gpt-4o"
    assert body["instructions"].startswith("You decide whether")
    assert body["input"] == PROMPT

    output_format = body["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert output_format["schema"]["additionalProperties"] is False
    # Strict mode requires every property to be required, defaults included.
    assert set(output_format["schema"]["required"]) == set(
        output_format["schema"]["properties"]
    )

    assert result.satisfied is True
    assert result.evidence_indices == [1]
    assert result.confidence == pytest.approx(0.82)


def test_openai_turns_an_empty_completion_into_a_readable_error(monkeypatch) -> None:
    """Hitting the token ceiling returns a 200 with nothing in it."""
    pytest.importorskip("openai")
    from resumatch.llm.client import LLMError, OpenAIClient

    empty = openai_response()
    empty["output"] = []
    empty["status"] = "incomplete"

    with Loopback(empty) as server:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
        monkeypatch.setenv("OPENAI_BASE_URL", f"{server.url}/v1")

        with pytest.raises(LLMError, match="no parseable output"):
            OpenAIClient().parse(
                instructions=INSTRUCTIONS, user=PROMPT, schema=SemanticVerdict
            )


# -- anthropic -------------------------------------------------------------


def test_anthropic_sends_the_request_the_messages_api_expects(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from resumatch.llm.client import AnthropicClient

    with Loopback(anthropic_response()) as server:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)

        result = AnthropicClient().parse(
            instructions=INSTRUCTIONS, user=PROMPT, schema=SemanticVerdict
        )

    body = server.request["body"]
    assert server.request["path"] == "/v1/messages"
    assert body["model"] == "claude-sonnet-5"
    assert body["system"].startswith("You decide whether")
    assert body["messages"] == [{"role": "user", "content": PROMPT}]
    assert body["max_tokens"] > 0

    output_format = body["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["additionalProperties"] is False

    assert result.satisfied is True
    assert result.evidence_indices == [1]
    assert result.confidence == pytest.approx(0.82)


def test_anthropic_turns_a_truncated_reply_into_a_readable_error(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from resumatch.llm.client import AnthropicClient, LLMError

    truncated = anthropic_response()
    truncated["content"] = []
    truncated["stop_reason"] = "max_tokens"

    with Loopback(truncated) as server:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)

        with pytest.raises(LLMError, match="no parseable output"):
            AnthropicClient().parse(
                instructions=INSTRUCTIONS, user=PROMPT, schema=SemanticVerdict
            )


# -- provider selection ----------------------------------------------------


def test_a_missing_key_is_an_error_naming_the_variable(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from resumatch.llm.client import AnthropicClient, LLMError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_claude_is_an_accepted_alias_for_anthropic(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from resumatch.llm.client import build_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    assert build_client("claude").name == "anthropic"


def test_the_key_present_decides_the_provider(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from resumatch.llm.client import build_client

    monkeypatch.delenv("RESUMATCH_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")

    assert build_client().name == "anthropic"
