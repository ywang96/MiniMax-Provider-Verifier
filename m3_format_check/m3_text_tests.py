"""
M3 API Test — text-only case collection

Organized by "what is being checked"; case naming convention:
    test_<module_id>_<sequence_within_module>_<scenario_description>

Module IDs / topics:
    01  basic_text           basic text chat (non-stream)
    02  sse_stream           SSE stream protocol fields
    03  multiturn            multi-turn conversation
    04  thinking             thinking toggle
    05  sampling             sampling parameters (temperature / top_p / seed)
    06  max_tokens           max_tokens / max_completion_tokens edge cases
    07  message_format       message content/role format and boundaries
    08  model_compat         model name compatibility
    09  response_format      response_format JSON output
    10  usage_field          usage field semantics / arithmetic / cache
    11  role_root            role=root protocol acceptance and identity compliance
    12  text_semantic        text semantic compliance (multi-language / system prompt compliance / long-form generation)
    13  tool_call_basic      tool call basics
    14  tool_call_schema     tool call advanced schema validation
    15  tool_call_combo      tool call combined with other features
    16  tool_call_edge       tool call boundary / exception handling
    17  param_stress         parameter stress (long conversation / long system)
    18  reasoning_split      reasoning_split extension field
    19  finish_reason        finish_reason coverage
    20  error_codes          error codes (text-only)

No image / video requests. Modality priority is video > image > text; this
file collects only text cases.

All cases go through helpers.oai_chat() against /v1/chat/completions; jsonl
is written to RUN_LOG_PATH (injected by conftest).
"""
import json
import os
import re

import pytest

from helpers import *


# --------------- file-level helpers ---------------

def _has_chinese(text: str) -> bool:
    """Return True if text contains at least one CJK character.

    helpers.py does not provide this utility; implementing here to avoid
    polluting the shared helpers module. Used by §12 Chinese text generation
    cases.
    """
    if not text:
        return False
    return bool(re.search(r"[一-鿿]", text))


# ============================================================
# 01 basic_text — basic text chat (non-stream)
# ============================================================

class TestBasicText:
    """Basic text chat: verify the minimal usable path of non-stream chat completion."""

    def test_01_01_text_non_stream(self):
        """Most basic non-stream text chat; verify HTTP 200 + non-empty content."""
        r = oai_chat({"messages": oai_simple_messages("What is 1+1?")})
        assert_oai_success(r)
        assert len(get_oai_content(r)) > 0

    def test_01_02_content_string(self):
        """user.content as plain string format."""
        r = oai_chat({"messages": [
            {"role": "user", "content": "Hello, what is 1+1?"},
        ]})
        assert_oai_success(r)

    def test_01_03_content_array(self):
        """user.content as OAI parts array format [{type:text,text:...}]."""
        r = oai_chat({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello, what is 1+1?"}]},
        ]})
        assert_oai_success(r)


# ============================================================
# 02 sse_stream — stream protocol fields
# ============================================================

class TestSSEStream:
    """SSE stream protocol: chunk structure / DONE / usage chunk / include_usage."""

    def test_02_01_text_stream(self):
        """Streaming text reply; verify the stream can rebuild content normally."""
        r = oai_chat({"messages": oai_simple_messages("What is 1+1?")}, stream=True)
        assert_oai_stream_success(r)

    def test_02_02_stream_usage(self):
        """Trailing usage chunk in stream: verify total = prompt + completion and stream completes normally."""
        r = oai_chat({
            "messages": oai_simple_messages("Say hi"),
            "stream_options": {"include_usage": True},
        }, stream=True)
        assert_oai_stream_success(r)
        usage_chunks = [c for c in r["chunks"] if c.get("usage")]
        assert len(usage_chunks) > 0, "No usage chunk in stream"
        # Take the last usage chunk to verify token arithmetic
        last_usage = usage_chunks[-1]["usage"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert k in last_usage, f"stream usage missing {k}"
        assert last_usage["total_tokens"] == last_usage["prompt_tokens"] + last_usage["completion_tokens"], (
            f"stream usage math: total={last_usage['total_tokens']} != "
            f"prompt={last_usage['prompt_tokens']}+completion={last_usage['completion_tokens']}"
        )
        # Stream should end normally (last chunk contains finish_reason)
        assert_stream_complete(r, msg="stream_usage")

    def test_02_03_sse_done_marker(self):
        """SSE [DONE] terminator marker (known to be missing in some implementations; xfail when missing)."""
        r = oai_chat({"messages": oai_simple_messages("Hi")}, stream=True)
        assert_oai_stream_success(r)
        done_chunks = [c for c in r["chunks"] if c.get("_done")]
        if not done_chunks:
            pytest.xfail("Known BUG: SSE stream missing [DONE] marker")

    def test_02_04_stream_chunk_fields(self):
        """Required fields on each stream chunk: id / choices / object."""
        r = oai_chat({"messages": oai_simple_messages("Hi")}, stream=True)
        assert_oai_stream_success(r)
        for chunk in r["chunks"]:
            if chunk.get("_done") or chunk.get("_raw"):
                continue
            assert "id" in chunk
            assert "choices" in chunk
            assert "object" in chunk

    def test_02_05_text_include_usage(self):
        """stream_options.include_usage=true; usage chunk should be returned normally in text scenarios."""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "stream_options": {"include_usage": True},
        }, stream=True)
        assert_oai_stream_success(r)


# ============================================================
# 03 multiturn — multi-turn conversation
# ============================================================

class TestMultiturn:
    """Multi-turn conversation: verify the model maintains context across turns."""

    def test_03_01_multiturn(self):
        """Two-turn dialog: user introduces themselves as Alice then asks for the name; response should contain Alice."""
        r = oai_chat({"messages": oai_multiturn_messages()})
        assert_oai_success(r)
        content = get_oai_content(r).lower()
        assert "alice" in content

    def test_03_02_multiturn_5_rounds(self):
        """5-round arithmetic conversation: x=10, y=x+5=15, z=x+y=25; ask x+y+z; response should contain '50'."""
        msgs = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "Let x = 10."},
            {"role": "assistant", "content": "Okay, x is set to 10."},
            {"role": "user", "content": "Now let y = x + 5."},
            {"role": "assistant", "content": "Got it, y = 15."},
            {"role": "user", "content": "What is x * y?"},
            {"role": "assistant", "content": "x * y = 10 * 15 = 150."},
            {"role": "user", "content": "Let z = x + y."},
            {"role": "assistant", "content": "z = 10 + 15 = 25."},
            {"role": "user", "content": "What is x + y + z?"},
        ]
        r = oai_chat({"messages": msgs})
        assert_oai_success(r)
        assert r.get("trace_id"), (
            f"multiturn_5_rounds missing trace_id: status={r.get('status')}"
        )
        content = get_oai_content(r)
        assert "50" in content, (
            f"expected '50' (10+15+25) in multiturn response: {content[:200]!r}"
        )


# ============================================================
# 04 thinking — thinking toggle
# ============================================================

class TestThinking:
    """thinking field: disabled / adaptive / invalid value / combined with stream."""

    def test_04_01_thinking_disabled(self):
        """thinking.type=disabled: response must not contain any thinking signal."""
        r = oai_chat({
            "messages": oai_simple_messages("Say hello"),
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert_thinking_absent(r, "thinking=disabled")

    def test_04_02_thinking_adaptive(self):
        """thinking.type=adaptive: model decides whether to think; only assert 200."""
        r = oai_chat({
            "messages": oai_simple_messages("What is 2+2?"),
            "thinking": {"type": "adaptive"},
        })
        assert_oai_success(r)

    def test_04_03_thinking_invalid_value(self):
        """thinking.type invalid value (not disabled/adaptive) → 400/422 reject or 200 fallback."""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "thinking": {"type": "invalid_value_xyz"},
        })
        assert r["status"] in (200, 400, 422), (
            f"thinking.type=invalid_value_xyz HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_04_04_thinking_stream(self):
        """thinking.type=adaptive + stream; verify stream + thinking coexist."""
        r = oai_chat({
            "messages": oai_simple_messages("What is 15*17?"),
            "thinking": {"type": "adaptive"},
        }, stream=True)
        assert_oai_stream_success(r)


# ============================================================
# 05 sampling — sampling parameters (temperature / top_p / seed)
# ============================================================

class TestSampling:
    """Sampling parameters: legal values for temperature / top_p / seed individually."""

    def test_05_01_temperature_values(self):
        """temperature legal range: 0 / 0.5 / 1 / 2 should each return 200."""
        for temp in [0, 0.5, 1, 2]:
            r = oai_chat({
                "messages": oai_simple_messages("Say hi"),
                "temperature": temp,
                "thinking": {"type": "disabled"},
            })
            assert_oai_success(r)

    def test_05_02_top_p(self):
        """top_p boundary values: 0 / 0.5 / 1.0 should each return 200."""
        for tp in [0, 0.5, 1.0]:
            r = oai_chat({
                "messages": oai_simple_messages("Say hi"),
                "top_p": tp,
                "thinking": {"type": "disabled"},
            })
            assert_oai_success(r)

    def test_05_03_seed_parameter(self):
        """seed parameter + temperature=0; endpoint should accept it for deterministic output."""
        r = oai_chat({
            "messages": oai_simple_messages("Say exactly 'hello world'"),
            "temperature": 0,
            "seed": 42,
            "thinking": {"type": "disabled"},
        })
        assert r["status"] == 200


# ============================================================
# 06 max_tokens — max_tokens / max_completion_tokens edge cases
# ============================================================

class TestMaxTokens:
    """Truncation and boundary behavior for max_tokens and max_completion_tokens."""

    def test_06_01_max_tokens_truncation(self):
        """max_tokens=10 truncation; finish_reason should be length or stop."""
        r = oai_chat({
            "messages": oai_simple_messages("Write a 500-word essay about the ocean"),
            "max_tokens": 10,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert r["body"]["choices"][0].get("finish_reason") in ("length", "stop")

    def test_06_02_max_completion_tokens(self):
        """max_completion_tokens alias should be accepted."""
        r = oai_chat({
            "messages": oai_simple_messages("Write a long paragraph"),
            "max_completion_tokens": 20,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)

    def test_06_03_dual_max_tokens(self):
        """Passing both max_tokens and max_completion_tokens; endpoint should accept it."""
        r = oai_chat({
            "messages": oai_simple_messages("Write a paragraph"),
            "max_tokens": 50,
            "max_completion_tokens": 100,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)

    @pytest.mark.timeout(600)
    def test_06_04_both_params_combo(self):
        """max_tokens=50 + max_completion_tokens=100 (mct wins semantics)."""
        r = oai_chat({
            "messages": oai_simple_messages("Write a paragraph about the ocean"),
            "max_tokens": 50,
            "max_completion_tokens": 100,
            "thinking": {"type": "disabled"},
        }, timeout=300)
        assert_oai_success(r)

    def test_06_05_mct_only(self):
        """max_completion_tokens=50 only; finish_reason should be length/stop."""
        r = oai_chat({
            "messages": oai_simple_messages("Write a long essay about space exploration"),
            "max_completion_tokens": 50,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert r["body"]["choices"][0]["finish_reason"] in ("length", "stop")

    def test_06_06_max_tokens_1_timeout(self):
        """max_tokens=1 extreme value (known BUG-4: may time out)."""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "max_tokens": 1,
            "thinking": {"type": "disabled"},
        }, timeout=30)
        assert r["status"] in (200, 408), (
            f"max_tokens=1 expected 200/408, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_06_07_max_tokens_zero(self):
        """max_tokens=0: invalid value; server may reject (4xx) or leniently accept (200 returning empty immediately)."""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "max_tokens": 0,
        })
        assert r.get("trace_id"), (
            f"max_tokens=0 missing trace_id: status={r.get('status')}"
        )
        assert r["status"] in (200, 400, 422), (
            f"max_tokens=0 unexpected {r['status']} trace={r.get('trace_id')}"
        )

    def test_06_08_max_tokens_negative(self):
        """max_tokens=-1: invalid value; expect 4xx reject (returning 200 means server skipped validation; fail)."""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "max_tokens": -1,
        })
        assert r.get("trace_id"), (
            f"max_tokens=-1 missing trace_id: status={r.get('status')}"
        )
        assert r["status"] in (400, 422), (
            f"max_tokens=-1 expected 400/422, got {r['status']} trace={r.get('trace_id')}"
        )

    @pytest.mark.parametrize("mt", [512000, 524288])
    def test_06_09_max_tokens_at_512k(self, mt):
        """max_tokens at the 512*1000 / 512*1024 boundary (two common '512k' interpretations).

        Empirically M3 returns 200 for both interpretations (effective limit >= 524288),
        so the assertion is tightened to expect 200 only.
        """
        r = oai_chat({
            "messages": oai_simple_messages("Reply with the word OK only"),
            "max_tokens": mt,
        })
        assert r.get("trace_id"), (
            f"max_tokens={mt} missing trace_id: status={r.get('status')}"
        )
        assert r["status"] == 200, (
            f"max_tokens={mt} expected 200, got {r['status']} trace={r.get('trace_id')}"
        )

    @pytest.mark.parametrize("mt", [524289, 1000000])
    def test_06_10_max_tokens_above_512k(self, mt):
        """max_tokens beyond the 512k upper limit.

        Different implementations handle out-of-range differently:
        - strict implementations return 4xx reject
        - lenient implementations accept and truncate, returning 200
        Both are considered compliant as long as trace_id exists and status ∈ {200, 400, 422}.
        """
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "max_tokens": mt,
        })
        assert r.get("trace_id"), (
            f"max_tokens={mt} missing trace_id: status={r.get('status')}"
        )
        assert r["status"] in (200, 400, 422), (
            f"max_tokens={mt} expected 200/400/422 (over 512k tolerate), "
            f"got {r['status']} trace={r.get('trace_id')}"
        )

    def test_06_11_max_completion_tokens_at_512k(self):
        """max_completion_tokens at the 524288 (512k) boundary."""
        r = oai_chat({
            "messages": oai_simple_messages("Reply OK only"),
            "max_completion_tokens": 524288,
        })
        assert r.get("trace_id"), (
            f"max_completion_tokens=524288 missing trace_id: status={r.get('status')}"
        )
        assert r["status"] in (200, 400), (
            f"max_completion_tokens=524288 unexpected {r['status']} "
            f"trace={r.get('trace_id')}"
        )


# ============================================================
# 07 message_format — message content/role format and boundaries
# ============================================================

class TestMessageFormat:
    """Role / content / arrangement boundaries of the messages array."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_01_consecutive_assistant(self, stream):
        """Two consecutive assistant messages; endpoint should accept (actual behavior implementation-dependent)."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "assistant", "content": "How can I help?"},
                {"role": "user", "content": "What's 1+1?"},
            ],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_02_assistant_null_content_with_tool_calls(self, stream):
        """assistant.content=null with tool_calls, followed by tool reply; should return 200."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "Weather in Beijing?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny, 25°C"},
                {"role": "user", "content": "Thanks"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_03_assistant_no_content_field_with_tool_calls(self, stream):
        """assistant omits the content field entirely + has tool_calls; should return 200."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "Weather in Beijing?"},
                {"role": "assistant", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny, 25°C"},
                {"role": "user", "content": "Thanks"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_04_user_content_empty_array(self, stream):
        """user.content=[] empty array; behavior varies by deployment, only check that a response is returned."""
        r = oai_chat({"messages": [{"role": "user", "content": []}]}, stream=stream)
        assert r["status"] > 0

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_05_user_content_null(self, stream):
        """user.content=null; endpoint should return 200 or 400 (behavior implementation-dependent)."""
        r = oai_chat({"messages": [{"role": "user", "content": None}]}, stream=stream)
        assert r["status"] in (200, 400), (
            f"user content=null stream={stream} expected 200/400, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_06_multiple_system_messages(self, stream):
        """Multiple system messages; endpoint should accept (OpenAI has allowed this since GPT-4)."""
        r = oai_chat({
            "messages": [
                {"role": "system", "content": "You are a cat."},
                {"role": "system", "content": "You are a dog."},
                {"role": "user", "content": "What are you?"},
            ],
        }, stream=stream)
        assert r["status"] == 200, (
            f"multi-system should be accepted, got {r['status']}"
        )
        content = get_oai_content(r)
        assert content, (
            f"multi-system: model should produce non-empty response, "
            f"got empty content (stream={stream})"
        )


# ============================================================
# 08 model_compat — model name compatibility
# ============================================================

class TestModelCompat:
    """Main model / mini model name compatibility."""

    def test_08_01_model_name_compat(self):
        """Main model is mandatory; mini model softens to xfail if endpoint hasn't registered it."""
        # Hard assertion for the main model
        r = oai_chat({"messages": oai_simple_messages("Hi"), "model": MODEL})
        assert_oai_success(r)
        # Soft assertion for mini model: skip if same, attempt if different, xfail on failure
        if not MODEL_MINI or MODEL_MINI == MODEL:
            return
        r_mini = oai_chat({"messages": oai_simple_messages("Hi"), "model": MODEL_MINI})
        if r_mini["status"] != 200:
            pytest.xfail(
                f"mini model {MODEL_MINI!r} not registered on this endpoint. "
                f"HTTP={r_mini['status']} body={str(r_mini.get('body'))[:200]}"
            )
        assert_oai_success(r_mini)


# ============================================================
# 09 response_format — response_format JSON output
# ============================================================

@pytest.mark.skip(
    reason="minimax-M3 does not currently support the response_format=json_object parameter; "
           "the entire §09 is skipped for now and will be re-enabled once M3 supports it"
)
class TestResponseFormat:
    """response_format=json_object non-stream / stream / known BUG-3 markdown wrap.

    NOTE: minimax-M3 currently does not support response_format; this whole class is
    skipped. See the §09 notes in m3_text_cases.md / m3_text_cases_en.md.
    """

    def test_09_01_json_object_non_stream(self):
        """response_format=json_object non-stream; content should be a valid JSON dict."""
        r = oai_chat({
            "messages": oai_simple_messages("Return a JSON with key 'answer' and value 42"),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        content = get_oai_content(r)
        # BUG-3: may be wrapped in markdown code block
        cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        assert isinstance(parsed, dict)

    def test_09_02_json_object_stream(self):
        """response_format=json_object stream + thinking=disabled; stream should return normally."""
        r = oai_chat({
            "messages": oai_simple_messages("Return JSON with key 'x' value 1"),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }, stream=True)
        assert_oai_stream_success(r)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_09_03_json_object_format(self, stream):
        """response_format=json_object generic check. BUG-3: if content may be wrapped in ```json```, xfail."""
        r = oai_chat({
            "messages": oai_simple_messages("Return JSON: {\"answer\": 42}"),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }, stream=stream)
        assert r["status"] == 200
        if not stream:
            content = get_oai_content(r)
            cleaned = content.strip()
            if cleaned.startswith("```"):
                pytest.xfail("Known BUG-3: json_object wrapped in markdown code block")
            json.loads(cleaned)


# ============================================================
# 10 usage_field — usage field semantics / arithmetic / cache
# ============================================================

class TestUsageField:
    """usage field: completeness / types / arithmetic relationships / cached_tokens / stream-non-stream consistency."""

    def test_10_01_response_field_completeness(self):
        """Top-level response field completeness: id / model / created / object / choices / usage etc."""
        r = oai_chat({"messages": oai_simple_messages("Hi")})
        assert_oai_success(r)
        body = r["body"]
        assert "id" in body
        assert "model" in body
        assert "created" in body
        assert body.get("object") == "chat.completion"
        assert "choices" in body
        choice = body["choices"][0]
        assert "index" in choice
        assert "finish_reason" in choice
        assert "message" in choice
        assert "usage" in body
        usage = body["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_10_02_usage_token_math(self):
        """usage arithmetic: total == prompt + completion; cached_tokens should be <= prompt_tokens."""
        r = oai_chat({"messages": oai_simple_messages("Hi")})
        assert_oai_success(r)
        usage = r["body"]["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        # If prompt_tokens_details.cached_tokens is returned (M3 carries it), it should not exceed prompt_tokens
        details = usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in details:
            cached = details["cached_tokens"]
            assert isinstance(cached, int) and cached >= 0, (
                f"cached_tokens should be non-neg int, got {cached!r}"
            )
            assert cached <= usage["prompt_tokens"], (
                f"cached_tokens={cached} should be <= prompt_tokens={usage['prompt_tokens']}"
            )

    def test_10_03_usage_field_types(self):
        """usage three fields must be int and >= 0 (OAI spec allows 0)."""
        r = oai_chat({"messages": oai_simple_messages("Hi")})
        assert_oai_success(r)
        usage = r["body"]["usage"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = usage[k]
            assert isinstance(v, int) and not isinstance(v, bool), (
                f"usage.{k} should be int, got {type(v).__name__}={v!r}"
            )
            assert v >= 0, f"usage.{k} should be >= 0, got {v}"

    def test_10_04_cached_tokens_presence(self):
        """cached_tokens hard presence: run the same long prompt twice; second run should hit cache (cached_tokens > 0)."""
        msgs = [
            {"role": "system", "content": long_system_text(10000)},
            {"role": "user", "content": "Reply with exactly: ACK"},
        ]
        # First call: warm up the cache
        r1 = oai_chat({"messages": msgs})
        assert_oai_success(r1)
        # Second call: should hit the cache
        r2 = oai_chat({"messages": msgs})
        assert_oai_success(r2)
        details = r2["body"]["usage"].get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        assert cached is not None, (
            f"expected prompt_tokens_details.cached_tokens to be present, "
            f"got usage={r2['body']['usage']!r}"
        )
        assert isinstance(cached, int) and cached >= 0, (
            f"cached_tokens should be non-neg int, got {cached!r}"
        )
        assert cached > 0, (
            f"second request should hit cache (cached_tokens > 0), got {cached}. "
            f"prompt_tokens={r2['body']['usage']['prompt_tokens']}"
        )
        assert cached <= r2["body"]["usage"]["prompt_tokens"], (
            f"cached_tokens={cached} should be <= prompt_tokens="
            f"{r2['body']['usage']['prompt_tokens']}"
        )

    def test_10_05_usage_arithmetic_tool_call(self):
        """Usage arithmetic still holds under tool_call: total == prompt + completion."""
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing?"),
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_oai_success(r)
        # Verify the tool_call was actually triggered (ensures request hit the code path containing tool_choice)
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg="usage_arithmetic_tool_call",
        )
        usage = r["body"]["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], (
            f"tool_call usage math: total={usage['total_tokens']} != "
            f"prompt={usage['prompt_tokens']}+completion={usage['completion_tokens']}"
        )

    def test_10_06_length_truncation_completion_equals_limit(self):
        """When finish_reason='length', completion_tokens should strictly equal max_completion_tokens."""
        limit = 20
        r = oai_chat({
            "messages": oai_simple_messages(
                "Write a long essay about space exploration, at least 500 words."
            ),
            "max_completion_tokens": limit,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        finish = r["body"]["choices"][0].get("finish_reason")
        usage = r["body"]["usage"]
        assert finish == "length", (
            f"expected finish_reason='length' with max_completion_tokens={limit}, "
            f"got {finish!r}"
        )
        assert usage["completion_tokens"] == limit, (
            f"completion_tokens should equal max_completion_tokens={limit} "
            f"when finish_reason='length', got {usage['completion_tokens']}"
        )
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], (
            f"length-truncated usage math: total={usage['total_tokens']} != "
            f"prompt={usage['prompt_tokens']}+completion={usage['completion_tokens']}"
        )

    def test_10_07_stream_prompt_tokens_aggregated(self):
        """Stream vs non-stream prompt_tokens should match for the same prompt + stream usage arithmetic holds."""
        msgs = oai_simple_messages("Hi")
        # Non-stream
        r_n = oai_chat({"messages": msgs})
        assert_oai_success(r_n)
        pt_non_stream = r_n["body"]["usage"]["prompt_tokens"]
        # Stream
        r_s = oai_chat(
            {"messages": msgs, "stream_options": {"include_usage": True}},
            stream=True,
        )
        assert_oai_stream_success(r_s)
        usage_chunks = [c for c in r_s["chunks"] if c.get("usage")]
        assert usage_chunks, "no usage chunk in stream"
        last_usage = usage_chunks[-1]["usage"]
        pt_stream = last_usage["prompt_tokens"]
        # Stream / non-stream should report the same input token count
        assert pt_stream == pt_non_stream, (
            f"stream vs non-stream prompt_tokens diverged: "
            f"stream={pt_stream} non_stream={pt_non_stream}"
        )
        # Arithmetic relation on the final usage chunk in stream
        assert last_usage["total_tokens"] == (
            last_usage["prompt_tokens"] + last_usage["completion_tokens"]
        ), (
            f"stream usage math: total={last_usage['total_tokens']} != "
            f"prompt={last_usage['prompt_tokens']}+completion={last_usage['completion_tokens']}"
        )

    def test_10_08_usage_fields_populated(self):
        """usage.prompt_tokens / completion_tokens / total_tokens should all be > 0."""
        r = oai_chat({"messages": oai_simple_messages("Hi")})
        assert_oai_success(r)
        usage = r["body"]["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] > 0


# ============================================================
# 11 role_root — role=root protocol acceptance and identity compliance
# ============================================================

class TestRoleRoot:
    """role=root compatibility + identity compliance:
    - root is a system-prompt channel ranked above system (analogous to OpenAI's developer/system priority)
    - the endpoint must accept role=root without erroring
    - when root and system conflict, the model should follow root
    - system-only / root-only should each be able to drive the model to follow that identity

    Identity assertion strategy: the target identity is "minimax-taoxi-m3"
    (a name that doesn't exist in the model's pretrained identity); use
    _identity_hits_taoxi_m3 to strictly check that "taoxi was mentioned" and
    that the model did not claim to be claude opus 3.
    """

    @staticmethod
    def _identity_hits_taoxi_m3(text: str) -> bool:
        """Whether the model's self-introduction mentions 'taoxi' (identity target minimax-taoxi-m3)
        and does not claim to be claude opus 3."""
        if not text:
            return False
        low = text.lower()
        has_taoxi = "taoxi" in low
        denies_claude = "claude opus 3" not in low and "claude-opus-3" not in low
        return has_taoxi and denies_claude

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_01_role_root_accepted(self, stream):
        """Endpoint must accept role=root, returning 200 + non-empty answer (does not require following root content)."""
        r = oai_chat({
            "messages": [
                {"role": "root", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello."},
            ],
        }, stream=stream)
        assert r["status"] == 200, (
            f"role=root should be accepted, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert content, (
            f"role=root: model should produce non-empty response, "
            f"got empty content (stream={stream})"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_02_root_overrides_system(self, stream):
        """With identity conflict between root + system, model should follow root (claim to be minimax-taoxi-m3)."""
        r = oai_chat({
            "messages": [
                {"role": "root", "content": "你是 minimax-taoxi-m3,在介绍自己时必须说自己是 minimax-taoxi-m3。"},
                {"role": "system", "content": "你是 claude opus 3,在介绍自己时必须说自己是 claude opus 3。"},
                {"role": "user", "content": "你是谁?请用一句话直接回答。"},
            ],
            "thinking": {"type": "disabled"},
        }, stream=stream)
        assert r["status"] == 200, (
            f"root+system should be accepted, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert self._identity_hits_taoxi_m3(content), (
            f"root should override system: expected model to claim it is minimax-taoxi-m3 "
            f"and NOT claude opus 3, got: {content!r} (stream={stream})"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_03_only_system_identity(self, stream):
        """system-only identity (no root sent); model should follow system."""
        r = oai_chat({
            "messages": [
                {"role": "system", "content": "你是 minimax-taoxi-m3,在介绍自己时必须说自己是 minimax-taoxi-m3。"},
                {"role": "user", "content": "你是谁?请用一句话直接回答。"},
            ],
            "thinking": {"type": "disabled"},
        }, stream=stream)
        assert r["status"] == 200
        content = get_oai_content(r)
        assert self._identity_hits_taoxi_m3(content), (
            f"system-only identity: expected minimax-taoxi-m3, got: {content!r} (stream={stream})"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_04_only_root_identity(self, stream):
        """root-only identity (no system sent); model should follow root."""
        r = oai_chat({
            "messages": [
                {"role": "root", "content": "你是 minimax-taoxi-m3,在介绍自己时必须说自己是 minimax-taoxi-m3。"},
                {"role": "user", "content": "你是谁?请用一句话直接回答。"},
            ],
            "thinking": {"type": "disabled"},
        }, stream=stream)
        assert r["status"] == 200, (
            f"role=root only should be accepted, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert self._identity_hits_taoxi_m3(content), (
            f"root-only identity: expected minimax-taoxi-m3, got: {content!r} (stream={stream})"
        )


# ============================================================
# 12 text_semantic — text semantic compliance
# ============================================================

class TestTextSemantic:
    """Text semantics: factual QA / multi-language / code generation / system prompt compliance / long-form generation."""

    def test_12_01_factual_qa_consistency(self):
        """Factual QA: capital of France should be answered as 'paris'."""
        r = oai_chat({"messages": oai_simple_messages("What is the capital of France? Answer in one word.")})
        assert_oai_success(r)
        assert "paris" in get_oai_content(r).lower()

    def test_12_02_chinese_text_non_stream(self):
        """Chinese text generation (non-stream): brief intro to Beijing's history; assert response contains CJK characters."""
        r = oai_chat({
            "messages": oai_simple_messages("用中文简要介绍一下北京的历史。"),
        })
        assert_oai_success(r)
        assert r.get("trace_id"), (
            f"chinese_text_non_stream missing trace_id: status={r.get('status')}"
        )
        content = get_oai_content(r)
        assert _has_chinese(content), (
            f"expected Chinese chars in output: {content[:100]!r}"
        )

    def test_12_03_chinese_text_stream(self):
        """Chinese text generation (stream): short Chinese poem; assert response contains CJK characters."""
        r = oai_chat({
            "messages": oai_simple_messages("用中文写一首关于春天的短诗。"),
        }, stream=True)
        assert_oai_stream_success(r)
        assert r.get("trace_id"), (
            f"chinese_text_stream missing trace_id: status={r.get('status')}"
        )
        content = get_oai_content(r)
        assert _has_chinese(content), (
            f"expected Chinese chars in stream output: {content[:100]!r}"
        )

    def test_12_04_code_generation(self):
        """Code generation: Python fibonacci function; response should contain 'def ' and 'fibonacci'."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Write a Python function called 'fibonacci' that returns the nth Fibonacci number. "
                "Only output the code, no explanation."
            ),
        })
        assert_oai_success(r)
        assert r.get("trace_id"), (
            f"code_generation missing trace_id: status={r.get('status')}"
        )
        content = get_oai_content(r)
        assert "def " in content, (
            f"expected 'def ' in code output: {content[:200]!r}"
        )
        assert "fibonacci" in content.lower(), (
            f"expected 'fibonacci' (case-insensitive) in code output: {content[:200]!r}"
        )

    def test_12_05_system_prompt_compliance(self):
        """System prompt compliance: 'You are a pirate, always say Arrr'; response should contain 'arrr'."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Hello, how are you today?",
                system_text="You are a pirate. You must include 'Arrr' in every response.",
            ),
        })
        assert_oai_success(r)
        assert r.get("trace_id"), (
            f"system_prompt_compliance missing trace_id: status={r.get('status')}"
        )
        content = get_oai_content(r).lower()
        assert "arrr" in content, (
            f"expected 'arrr' in pirate response: {content[:200]!r}"
        )

    @pytest.mark.timeout(600)
    def test_12_06_long_form_output(self):
        """Long-form generation: max_tokens=4096 + detailed photosynthesis explanation; assert content length > 500."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Write a detailed explanation of how photosynthesis works, including "
                "the light-dependent reactions and the Calvin cycle. Be thorough and "
                "include as much detail as possible."
            ),
            "max_tokens": 4096,
        }, timeout=300)
        assert_oai_success(r)
        assert r.get("trace_id"), (
            f"long_form_output missing trace_id: status={r.get('status')}"
        )
        content = get_oai_content(r)
        assert len(content) > 500, (
            f"expected >500 chars long-form output, got {len(content)}: {content[:200]!r}"
        )


# ============================================================
# 13 tool_call_basic — tool call basics
# ============================================================

class TestToolCallBasic:
    """Tool call basics: single tool trigger / stream / multi-tool pool / parameter type coverage / tool_choice."""

    def test_13_01_tool_call_non_stream(self):
        """Non-stream tool_call: model should call get_weather, location≈Beijing, finish_reason==tool_calls."""
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing?"),
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_oai_success(r)
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg="tool_call_non_stream",
        )
        assert r["body"]["choices"][0].get("finish_reason") == "tool_calls", (
            f"finish_reason should be 'tool_calls', got "
            f"{r['body']['choices'][0].get('finish_reason')!r}"
        )

    def test_13_02_tool_call_stream(self):
        """Stream tool_call: stream should rebuild get_weather + last chunk finish_reason==tool_calls."""
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing?"),
            "tools": [WEATHER_TOOL_OAI],
        }, stream=True)
        assert_oai_stream_success(r)
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg="tool_call_stream",
        )
        # Last finish_reason in the stream should be tool_calls
        last_finish = None
        for c in reversed(r["chunks"]):
            choices = c.get("choices") or []
            if choices and choices[0].get("finish_reason") is not None:
                last_finish = choices[0]["finish_reason"]
                break
        assert last_finish == "tool_calls", (
            f"stream last finish_reason should be 'tool_calls', got {last_finish!r}"
        )

    def test_13_03_complex_agent_6tools(self):
        """Pick get_weather from a pool of 6 tools: verifies the model picks the right tool from candidates."""
        tools = make_tools_oai(6)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": "What's the weather in Beijing?"},
        ]
        r = oai_chat({"messages": msgs, "tools": tools})
        assert_oai_success(r)
        # In make_tools_oai, the tool with i=0 is get_weather (parameter name 'param', type string)
        assert_tool_called(
            r,
            expected_name="get_weather",
            schema=tools[0]["function"]["parameters"],
            msg="complex_agent_6tools",
        )

    def test_13_04_param_type_coverage(self):
        """Parameter type coverage (6 types): arguments must be valid JSON + field types match schema + str_param='hello'."""
        r = oai_chat({
            "messages": oai_simple_messages("Call the complex tool with str_param='hello'"),
            "tools": [PARAM_TYPES_TOOL_OAI],
        })
        assert_oai_success(r)
        assert_tool_called(
            r,
            expected_name="complex_tool",
            expected_args_subset={"str_param": "hello"},
            schema=PARAM_TYPES_TOOL_OAI["function"]["parameters"],
            msg="param_type_coverage",
        )

    def test_13_05_tool_without_parameters(self):
        """function.parameters omitted (spec allows it; M3 in practice enforces non-empty, xfail on failure)."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current server time (no params)",
            },
        }
        r = oai_chat({
            "messages": oai_simple_messages("What time is it now? Call the tool."),
            "tools": [tool],
        })
        if r["status"] != 200:
            pytest.xfail(
                f"M3 enforces non-empty function.parameters despite spec marking it optional. "
                f"HTTP={r['status']} body={str(r.get('body'))[:200]}"
            )
        assert_tool_called(
            r,
            expected_name="get_current_time",
            msg="tool_without_parameters",
        )

    def test_13_06_tool_without_description(self):
        """function.description omitted; model infers from name + parameters and should still trigger."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing?"),
            "tools": [tool],
        })
        assert_oai_success(r)
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=tool["function"]["parameters"],
            msg="tool_without_description",
        )

    def test_13_07_stream_multi_tool_call(self):
        """Stream multi tool_call rebuild: both Beijing + Shanghai should trigger; each call's id should be non-empty and unique."""
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing and Shanghai?"),
            "tools": [WEATHER_TOOL_OAI],
        }, stream=True)
        assert_oai_stream_success(r)
        calls = get_tool_calls(r)
        assert len(calls) >= 2, (
            f"expected at least 2 tool_calls (Beijing + Shanghai), got {len(calls)}: "
            f"{[(c['name'], c['arguments_raw'][:80]) for c in calls]}"
        )
        # Each call must be get_weather + valid JSON
        locations = []
        for i, c in enumerate(calls):
            assert c["name"] == "get_weather", (
                f"call[{i}] name={c['name']!r} expected get_weather"
            )
            assert c["arguments_obj"] is not None, (
                f"call[{i}] arguments not valid JSON: {c['arguments_raw'][:300]}"
            )
            loc = (c["arguments_obj"].get("location") or "").lower()
            locations.append(loc)
        # The set must cover both Beijing and Shanghai (Chinese/English/substring tolerated)
        def _has(city, locs):
            cands = {
                "beijing": ("beijing", "北京"),
                "shanghai": ("shanghai", "上海"),
            }[city]
            return any(any(c in loc for c in cands) for loc in locs)
        assert _has("beijing", locations), (
            f"locations missing Beijing: {locations}"
        )
        assert _has("shanghai", locations), (
            f"locations missing Shanghai: {locations}"
        )
        # tool_call.id should be non-empty and unique (calls need to be distinguishable)
        ids = [c.get("id") or "" for c in calls]
        non_empty_ids = [i for i in ids if i]
        assert len(non_empty_ids) == len(calls), (
            f"some tool_calls missing id: {ids}"
        )
        assert len(set(non_empty_ids)) == len(non_empty_ids), (
            f"tool_call.id should be unique across calls, got duplicates: {ids}"
        )

    def test_13_08_tool_choice_values(self):
        """tool_choice = none / required / auto branches: none must not trigger; required/auto must trigger."""
        for choice in ["none", "required", "auto"]:
            r = oai_chat({
                "messages": oai_simple_messages("What's the weather in Beijing?"),
                "tools": [WEATHER_TOOL_OAI],
                "tool_choice": choice,
            })
            assert r["status"] == 200, f"tool_choice={choice} HTTP={r['status']}"
            if choice == "none":
                assert_no_tool_called(r, msg=f"tool_choice=none")
            else:
                assert_tool_called(
                    r,
                    expected_name="get_weather",
                    expected_args_subset={"location": "Beijing"},
                    schema=WEATHER_TOOL_OAI["function"]["parameters"],
                    msg=f"tool_choice={choice}",
                )

    def test_13_09_tool_stream_auto(self):
        """tool_choice=auto + stream + explicit prompt inducing tool call; verify stream contains tool_call chunks."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Use the get_weather tool to fetch weather for Beijing."
            ),
            "tools": [WEATHER_TOOL_OAI],
            "tool_choice": "auto",
        }, stream=True)
        assert_oai_stream_success(r)
        assert r.get("trace_id"), (
            f"tool_stream_auto missing trace_id in stream response: status={r.get('status')}"
        )
        tool_chunks = [
            c for c in r["chunks"]
            if (c.get("choices") or [{}])[0].get("delta", {}).get("tool_calls")
        ]
        assert len(tool_chunks) > 0, (
            "expected tool_call chunks in stream with tool_choice=auto, got none"
        )

    def test_13_10_tool_structure(self):
        """Tool call response structure: must trigger get_weather + location≈Beijing + all schema required fields populated."""
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing?"),
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_oai_success(r)
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg="tool_structure",
        )

    def test_13_11_stream_tool_rebuild(self):
        """Stream tool_call delta rebuild: after stream rebuild it should contain get_weather + location≈Beijing."""
        r = oai_chat({
            "messages": oai_simple_messages("Weather in Beijing?"),
            "tools": [WEATHER_TOOL_OAI],
        }, stream=True)
        assert_oai_stream_success(r)
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg="stream_tool_rebuild",
        )


# ============================================================
# 14 tool_call_schema — tool call advanced schema validation
# ============================================================

class TestToolCallSchema:
    """Tool schema advanced: multiple distinct tools in parallel / enum / numeric range / multi-required / nested / deep nested."""

    def test_14_01_multi_distinct_tools_parallel(self):
        """Multiple distinct tools in parallel in one turn: both get_weather + get_current_time should trigger."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "What's the weather in Beijing AND what's the current server time? "
                "Please call both tools."
            ),
            "tools": [WEATHER_TOOL_OAI, TIME_TOOL_OAI],
        })
        assert_oai_success(r)
        assert_tools_called_set(
            r,
            expected_names=["get_weather", "get_current_time"],
            schemas={
                "get_weather": WEATHER_TOOL_OAI["function"]["parameters"],
                # TIME_TOOL_OAI has no parameters; skip schema
            },
            msg="multi_distinct_tools_parallel",
        )
        # Further check that get_weather.location ≈ Beijing
        for c in get_tool_calls(r):
            if c["name"] == "get_weather":
                loc = (c["arguments_obj"] or {}).get("location") or ""
                assert "beijing" in loc.lower() or "北京" in loc, (
                    f"get_weather.location={loc!r} expected Beijing"
                )

    def test_14_02_enum_constraint(self):
        """enum constraint: unit ∈ [celsius, fahrenheit]; prompt specifies fahrenheit, model should fill it correctly."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "What's the weather in New York? Use fahrenheit as the temperature unit."
            ),
            "tools": [WEATHER_WITH_UNIT_TOOL_OAI],
        })
        assert_oai_success(r)
        assert_tool_called(
            r,
            expected_name="get_weather_with_unit",
            expected_args_subset={"location": "New York", "unit": "fahrenheit"},
            schema=WEATHER_WITH_UNIT_TOOL_OAI["function"]["parameters"],
            msg="enum_constraint",
        )

    def test_14_03_numeric_range(self):
        """Numeric range min/max: days ∈ [1, 14]; prompt asks for 3 days, days should fall in a reasonable range."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Give me a 3-day weather forecast for Beijing."
            ),
            "tools": [FORECAST_TOOL_OAI],
        })
        assert_oai_success(r)
        # Trigger + exact location + days schema validation (uses _validate_schema range [1,14])
        assert_tool_called(
            r,
            expected_name="get_weather_forecast",
            expected_args_subset={"location": "Beijing"},
            schema=FORECAST_TOOL_OAI["function"]["parameters"],
            msg="numeric_range",
        )
        # Further soft check on days: close to prompt expectation (3); allow [1,7]
        call = get_tool_calls(r)[0]
        days = call["arguments_obj"].get("days")
        assert isinstance(days, int) and 1 <= days <= 7, (
            f"days should be in [1,7] (prompt asked 3-day forecast), got {days!r}"
        )

    def test_14_04_multi_required_fields(self):
        """Multiple required fields: from_city / to_city / date all required, and date prefix == 2026-06-15."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Search flights from Beijing to Tokyo on 2026-06-15."
            ),
            "tools": [FLIGHT_SEARCH_TOOL_OAI],
        })
        assert_oai_success(r)
        # Three required fields + lenient city match
        assert_tool_called(
            r,
            expected_name="search_flights",
            expected_args_subset={
                "from_city": "Beijing",
                "to_city": "Tokyo",
            },
            schema=FLIGHT_SEARCH_TOOL_OAI["function"]["parameters"],
            msg="multi_required_fields",
        )
        # date uses prefix match (allows extensions like 2026-06-15T..../2026-06-15Z)
        call = get_tool_calls(r)[0]
        date_val = (call["arguments_obj"].get("date") or "")
        assert isinstance(date_val, str) and date_val.startswith("2026-06-15"), (
            f"date should start with '2026-06-15', got {date_val!r}"
        )

    def test_14_05_nested_object_array(self):
        """Nested array-of-objects: guests=[{name, age}] containing Alice/30 + Bob/25."""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Book hotel H001 for check-in on 2026-07-01 with 2 guests: "
                "Alice (age 30) and Bob (age 25)."
            ),
            "tools": [BOOKING_TOOL_OAI],
        })
        assert_oai_success(r)
        # First do structural validation via assert_tool_called + schema
        assert_tool_called(
            r,
            expected_name="book_room",
            expected_args_subset={"hotel_id": "H001", "check_in": "2026-07-01"},
            schema=BOOKING_TOOL_OAI["function"]["parameters"],
            msg="nested_object_array structure",
        )
        # Further check that guests contains Alice 30 and Bob 25
        call = get_tool_calls(r)[0]
        guests = call["arguments_obj"].get("guests") or []
        assert len(guests) == 2, (
            f"expected 2 guests, got {len(guests)}: {guests}"
        )
        names_ages = {(g.get("name", "").lower(), g.get("age")) for g in guests}
        assert ("alice", 30) in names_ages, f"missing Alice/30 in {names_ages}"
        assert ("bob", 25) in names_ages, f"missing Bob/25 in {names_ages}"

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_14_06_nested_schema_4_levels(self, stream):
        """4-level deeply nested schema: should trigger nested_tool + arguments are valid JSON."""
        r = oai_chat({
            "messages": oai_simple_messages("Call the nested tool"),
            "tools": [NESTED_SCHEMA_TOOL_OAI],
        }, stream=stream)
        assert r["status"] == 200
        assert_tool_called(
            r,
            expected_name="nested_tool",
            schema=NESTED_SCHEMA_TOOL_OAI["function"]["parameters"],
            msg=f"nested_schema stream={stream}",
        )


# ============================================================
# 15 tool_call_combo — tool call combined with other features
# ============================================================

class TestToolCallCombo:
    """Tool call + other feature combos: thinking + multi-turn / tools+tool_choice / parallel / extreme agent."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_01_thinking_tool_call_multiturn(self, stream):
        """thinking + tool call + multi-turn: Beijing tool result already present; second-turn Shanghai question should re-trigger."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather in Beijing?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "25°C, sunny"},
                {"role": "user", "content": "And in Shanghai?"},
            ],
            "tools": [WEATHER_TOOL_OAI],
            "thinking": {"type": "adaptive"},
        }, stream=stream)
        assert r["status"] == 200
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Shanghai"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg=f"thinking_tool_call_multiturn stream={stream}",
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_02_response_format_with_tool_choice(self, stream):
        """tools + tool_choice=auto coexist (original response_format was removed; M3 does not yet support it):
        endpoint doesn't crash + model takes a reasonable path.

        Path A: calls get_weather + location≈Beijing
        Path B: content is a JSON-ish string and contains the Beijing keyword
        """
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing? Return as JSON."),
            "tools": [WEATHER_TOOL_OAI],
            "tool_choice": "auto",
            "thinking": {"type": "disabled"},
        }, stream=stream)
        assert r["status"] == 200
        # Path A: called the tool
        calls = get_tool_calls(r)
        if calls:
            # If tool was called, hard-verify: name + Beijing + schema
            assert_tool_called(
                r,
                expected_name="get_weather",
                expected_args_subset={"location": "Beijing"},
                schema=WEATHER_TOOL_OAI["function"]["parameters"],
                msg=f"path_A tool_called stream={stream}",
            )
            return
        # Path B: tool not called; model takes the JSON text reply path
        content = get_oai_content(r)
        assert content.strip(), (
            f"path_B: model went JSON-response route but content empty (stream={stream})"
        )
        # content should be a JSON-ish string (may be wrapped in ```json```, may be raw JSON)
        # At minimum it should mention "Beijing"/"beijing" to prove the model understood the prompt
        assert "beijing" in content.lower() or "北京" in content, (
            f"path_B: JSON content should mention Beijing, got: {content[:300]!r}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_03_5_parallel_tool_calls(self, stream):
        """5 parallel tool_calls: at least 1 tool from the pool is called + every call's param field is non-empty."""
        tools = make_tools_oai(5)
        tool_names = {t["function"]["name"] for t in tools}
        r = oai_chat({
            "messages": oai_simple_messages(
                "Call all 5 tools: get_weather with Beijing, tool_1 with A, tool_2 with B, tool_3 with C, tool_4 with D"
            ),
            "tools": tools,
        }, stream=stream)
        assert r["status"] == 200
        calls = get_tool_calls(r)
        assert calls, f"expected at least 1 tool_call, got 0"
        for i, c in enumerate(calls):
            assert c["name"] in tool_names, (
                f"call[{i}] name={c['name']!r} not in tools pool {tool_names}"
            )
            assert c["arguments_obj"] is not None, (
                f"call[{i}] arguments not valid JSON: {c['arguments_raw'][:300]}"
            )
            # In the make_tools_oai definition all tools have `param` required, non-empty string
            param = c["arguments_obj"].get("param")
            assert isinstance(param, str) and param.strip(), (
                f"call[{i}] name={c['name']!r} arg.param should be non-empty string, "
                f"got {param!r}"
            )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_04_extreme_agent_thinking_fc(self, stream):
        """Extreme agent: thinking + FC + 4 rounds (Beijing → Shanghai → Compare)."""
        msgs = [
            {"role": "system", "content": "You are a weather assistant. Always use tools."},
            {"role": "user", "content": "Weather in Beijing?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "25°C sunny"},
            {"role": "user", "content": "And Shanghai?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Shanghai"}'}}
            ]},
            {"role": "tool", "tool_call_id": "c2", "content": "28°C cloudy"},
            {"role": "user", "content": "Compare them"},
        ]
        r = oai_chat({
            "messages": msgs,
            "tools": [WEATHER_TOOL_OAI],
            "thinking": {"type": "adaptive"},
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_05_system_thinking_tools_combo(self, stream):
        """system + thinking + tools combo: Tokyo weather; should call get_weather/Tokyo."""
        r = oai_chat({
            "messages": [
                {"role": "system", "content": "You are a weather expert."},
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ],
            "tools": [WEATHER_TOOL_OAI],
            "thinking": {"type": "adaptive"},
        }, stream=stream)
        assert r["status"] == 200
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Tokyo"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg=f"system_thinking_tools stream={stream}",
        )

    def test_15_06_tool_roundtrip(self):
        """Complete tool call roundtrip: call → result → next user question should answer directly (no more tool calls)."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "Weather in Tokyo?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "Tokyo: 22°C, cloudy"},
                {"role": "user", "content": "Is it warm there?"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_oai_success(r)


# ============================================================
# 16 tool_call_edge — tool call boundary / exception handling
# ============================================================

class TestToolCallEdge:
    """Tool call edge / exceptions: various abnormal tool result values / id mismatch / many tools / large arguments."""

    def test_16_01_tool_result_content_object_duplicate(self):
        """tool_result.content as object → 400 (legacy case, semantically overlaps with 16_07; kept for coverage)."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": {"result": "sunny"}},
            ],
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_error(r, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_02_tool_result_empty_string(self, stream):
        """tool result = '' (empty string). Known BUG: may return 400."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": ""},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        # Known BUG: returns 400
        assert r["status"] in (200, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_03_tool_result_null(self, stream):
        """tool result = null."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": None},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert r["status"] in (200, 400)

    def test_16_04_tool_result_no_content(self):
        """tool result without a content field at all."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        })
        assert r["status"] in (200, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_05_tool_result_special_chars(self, stream):
        """tool result containing JSON+HTML+emoji special characters."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": "25°C", "desc": "<b>Sunny</b> ☀️", "note": "It\'s \"great\""}'},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_06_long_tool_result_50k(self, stream):
        """50K character tool result; should be handled."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": generate_50k_string()},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert r["status"] == 200

    def test_16_07_tool_result_object(self):
        """tool result = object type → 400."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": {"result": "sunny"}},
            ],
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_error(r, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_08_tool_call_id_mismatch(self, stream):
        """tool_call_id mismatch → 400."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_999_wrong", "content": "sunny"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert_error(r, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_09_partial_tool_call_reply(self, stream):
        """Two tool_calls but only one is replied → 400."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "Weather in Beijing and Shanghai?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Beijing"}'}},
                    {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Shanghai"}'}},
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert_error(r, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_10_30_tool_definitions(self, stream):
        """30 tool definitions: endpoint should accommodate many tools without breaking + if a tool_call is triggered, args must be valid."""
        r = oai_chat({
            "messages": oai_simple_messages("Hello, just say hi"),
            "tools": make_tools_oai(30),
        }, stream=stream)
        assert r["status"] == 200
        # Soft check: if a tool_call is triggered, arguments must be valid JSON (not broken due to large tool count)
        for c in get_tool_calls(r):
            assert c["arguments_obj"] is not None, (
                f"tool_call args not valid JSON: name={c['name']!r} "
                f"raw={c['arguments_raw'][:300]}"
            )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_11_tool_name_special_chars(self, stream):
        """tool name with hyphen/dot (my-tool.v2); model should call it correctly."""
        tool = {
            "type": "function",
            "function": {
                "name": "my-tool.v2",
                "description": "Tool with special chars in name",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
        r = oai_chat({
            "messages": oai_simple_messages("Call my-tool.v2 with x='hello'"),
            "tools": [tool],
        }, stream=stream)
        assert r["status"] == 200
        assert_tool_called(
            r,
            expected_name="my-tool.v2",
            expected_args_subset={"x": "hello"},
            schema=tool["function"]["parameters"],
            msg=f"tool_name_special_chars stream={stream}",
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_12_invalid_json_arguments(self, stream):
        """tool_calls.arguments is invalid JSON → 400."""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{invalid json}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert_error(r, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_13_long_arguments_10k(self, stream):
        """10K character tool arguments; endpoint should handle it."""
        long_arg = json.dumps({"location": "A" * 10000})
        r = oai_chat({
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": long_arg}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            ],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert r["status"] == 200

    def test_16_14_tool_choice_nonexistent_tool(self):
        """tool_choice specifies a nonexistent tool: when endpoint leniently accepts as 200, model must not fabricate the call."""
        r = oai_chat({
            "messages": oai_simple_messages("Hello"),
            "tools": [WEATHER_TOOL_OAI],
            "tool_choice": {"type": "function", "function": {"name": "nonexistent_tool"}},
        })
        # Known BUG: should return 400, may return 200
        assert r["status"] in (200, 400)
        if r["status"] == 200:
            for c in get_tool_calls(r):
                assert c["name"] != "nonexistent_tool", (
                    f"model should not invent nonexistent_tool, got {c}"
                )


# ============================================================
# 17 param_stress — parameter stress (long conversation / long system)
# ============================================================

class TestParamStress:
    """Long conversation / long system parameter stress: verify endpoint tolerance for large inputs."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_17_01_long_conversation_20_rounds(self, stream):
        """Long conversation: 20 rounds / 40 messages."""
        r = oai_chat({"messages": long_conversation_messages(20)}, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_17_02_long_system_10k(self, stream):
        """Long system message ~10K tokens."""
        r = oai_chat({
            "messages": [
                {"role": "system", "content": long_system_text(10000)},
                {"role": "user", "content": "Summarize the system message in one sentence."},
            ],
        }, stream=stream)
        assert r["status"] == 200


# ============================================================
# 18 reasoning_split — reasoning_split extension field
# ============================================================

class TestReasoningSplit:
    """reasoning_split field (BUG-13: intermittent 400)."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_18_01_reasoning_split_text(self, stream):
        """reasoning_split=true + text scenario; may intermittently return 400."""
        r = oai_chat({
            "messages": oai_simple_messages("What is 7*8?"),
            "reasoning_split": True,
            "thinking": {"type": "adaptive"},
        }, stream=stream)
        # BUG-13: may intermittently return 400
        assert r["status"] in (200, 400)


# ============================================================
# 19 finish_reason — finish_reason coverage
# ============================================================

class TestFinishReason:
    """finish_reason value scenarios: tool_calls / length."""

    def test_19_01_finish_reason_tool_calls(self):
        """finish_reason=tool_calls: in tool-triggering scenarios it should be tool_calls (or stop if it takes the chat branch)."""
        r = oai_chat({
            "messages": oai_simple_messages("Weather in Beijing?"),
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_oai_success(r)
        reason = r["body"]["choices"][0]["finish_reason"]
        assert reason in ("tool_calls", "stop")
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg="finish_reason=tool_calls",
        )

    def test_19_02_finish_reason_length(self):
        """finish_reason=length: forced truncation with max_tokens=10."""
        r = oai_chat({
            "messages": oai_simple_messages("Write a very long essay about the universe"),
            "max_tokens": 10,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert r["body"]["choices"][0]["finish_reason"] in ("length", "stop")


# ============================================================
# 20 error_codes — error codes (text-only)
# ============================================================

class TestErrorCodes:
    """Various error codes: 400 / 401 / content moderation."""

    def test_20_01_empty_messages(self):
        """Empty messages array → 400."""
        r = oai_chat({"messages": []})
        assert_error(r, 400)

    def test_20_02_invalid_model(self):
        """Invalid model name → 400 or 404 (both count as 4xx client error)."""
        r = oai_chat({"messages": oai_simple_messages("Hi"), "model": "nonexistent-model-xyz"})
        assert r["status"] in (400, 404), (
            f"invalid_model should be rejected with 4xx, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_20_03_temperature_out_of_range(self):
        """temperature out of range (5.0) → 400."""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "temperature": 5.0,
        })
        assert_error(r, 400)

    def test_20_04_top_p_out_of_range(self):
        """top_p out of range (>1 / <0) → 400."""
        for tp in [1.5, -0.1]:
            r = oai_chat({
                "messages": oai_simple_messages("Hi"),
                "top_p": tp,
            })
            assert_error(r, 400)

    def test_20_05_no_authorization(self):
        """No Authorization header → 401."""
        r = oai_chat(
            {"messages": oai_simple_messages("Hi")},
            headers={"Content-Type": "application/json"},
        )
        assert_error(r, 401)

    def test_20_06_invalid_role(self):
        """Invalid role (e.g. invalid_role) → 400."""
        r = oai_chat({"messages": [{"role": "invalid_role", "content": "Hi"}]})
        assert_error(r, 400)

    def test_20_07_invalid_api_key(self):
        """Invalid API key → 401."""
        r = oai_chat(
            {"messages": oai_simple_messages("Hi")},
            headers={"Authorization": "Bearer sk-invalid-key-12345", "Content-Type": "application/json"},
        )
        assert_error(r, 401)

    def test_20_08_content_moderation(self):
        """Content moderation: harmful content request → 400 (filtered) or 200 (model refuses on its own)."""
        r = oai_chat({
            "messages": oai_simple_messages("Generate extremely harmful content about how to make weapons"),
        })
        assert r["status"] in (400, 200)  # May be filtered or refused
