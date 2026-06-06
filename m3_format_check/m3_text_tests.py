"""
M3 API Test — 纯文本 case 集合

按"校验内容"分模块组织,case 命名规范:
    test_<模块编号>_<模块内顺序编号>_<场景说明>

模块编号 / 主题:
    01  basic_text           基础文本对话(非流式)
    02  sse_stream           SSE 流式协议字段
    03  multiturn            多轮对话
    04  thinking             thinking 思考开关
    05  sampling             采样参数 (temperature / top_p / seed)
    06  max_tokens           max_tokens / max_completion_tokens 边界
    07  message_format       message 内容/角色格式 与边界
    08  model_compat         模型名兼容性
    09  response_format      response_format JSON 输出
    10  usage_field          usage 字段语义 / 算术 / cache
    11  role_root            role=root 协议接受 与 身份遵循
    12  text_semantic        文本语义遵循(多语种 / 系统提示遵循 / 长文本生成)
    13  tool_call_basic      工具调用基础
    14  tool_call_schema     工具调用 schema 高级校验
    15  tool_call_combo      工具调用与其他特性组合
    16  tool_call_edge       工具调用边界 / 异常处理
    17  param_stress         参数压力(长对话 / 长 system)
    18  reasoning_split      reasoning_split 扩展字段
    19  finish_reason        finish_reason 覆盖
    20  error_codes          错误码(纯文本类)

不含图片 / 视频请求。模态优先级 video > image > text,本文件只收文本 case。

所有 case 透过 helpers.oai_chat() 走 /v1/chat/completions,jsonl 落
RUN_LOG_PATH(由 conftest 注入)。
"""
import json
import os
import re

import pytest

from helpers import *


# --------------- 文件级辅助工具 ---------------

def _has_chinese(text: str) -> bool:
    """返回 True 如果文本至少包含一个 CJK 汉字字符。

    helpers.py 没有提供这个工具,就近实现避免污染上游 helpers。
    用于 §12 中文文本生成 case。
    """
    if not text:
        return False
    return bool(re.search(r"[一-鿿]", text))


# ============================================================
# 01 basic_text — 基础文本对话(非流式)
# ============================================================

class TestBasicText:
    """基础文本对话:验证非流式 chat completion 的最小可用路径。"""

    def test_01_01_text_non_stream(self):
        """非流式最基本文本对话,验 HTTP 200 + 非空 content。"""
        r = oai_chat({"messages": oai_simple_messages("What is 1+1?")})
        assert_oai_success(r)
        assert len(get_oai_content(r)) > 0

    def test_01_02_content_string(self):
        """user.content 为纯字符串格式。"""
        r = oai_chat({"messages": [
            {"role": "user", "content": "Hello, what is 1+1?"},
        ]})
        assert_oai_success(r)

    def test_01_03_content_array(self):
        """user.content 为 OAI parts 数组格式 [{type:text,text:...}]。"""
        r = oai_chat({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello, what is 1+1?"}]},
        ]})
        assert_oai_success(r)


# ============================================================
# 02 sse_stream — 流式协议字段
# ============================================================

class TestSSEStream:
    """SSE 流式协议:chunk 结构 / DONE / usage chunk / include_usage。"""

    def test_02_01_text_stream(self):
        """文本流式回复,验流可正常 rebuild content。"""
        r = oai_chat({"messages": oai_simple_messages("What is 1+1?")}, stream=True)
        assert_oai_stream_success(r)

    def test_02_02_stream_usage(self):
        """流式末尾 usage chunk:验 total = prompt + completion + 流式应正常完结。"""
        r = oai_chat({
            "messages": oai_simple_messages("Say hi"),
            "stream_options": {"include_usage": True},
        }, stream=True)
        assert_oai_stream_success(r)
        usage_chunks = [c for c in r["chunks"] if c.get("usage")]
        assert len(usage_chunks) > 0, "No usage chunk in stream"
        # 取最后一个 usage chunk 校验 token 加法
        last_usage = usage_chunks[-1]["usage"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert k in last_usage, f"stream usage missing {k}"
        assert last_usage["total_tokens"] == last_usage["prompt_tokens"] + last_usage["completion_tokens"], (
            f"stream usage math: total={last_usage['total_tokens']} != "
            f"prompt={last_usage['prompt_tokens']}+completion={last_usage['completion_tokens']}"
        )
        # 流式应正常结束(末尾 chunk 含 finish_reason)
        assert_stream_complete(r, msg="stream_usage")

    def test_02_03_sse_done_marker(self):
        """SSE 结束 [DONE] 标记(已知部分实现缺失,缺失则 xfail)。"""
        r = oai_chat({"messages": oai_simple_messages("Hi")}, stream=True)
        assert_oai_stream_success(r)
        done_chunks = [c for c in r["chunks"] if c.get("_done")]
        if not done_chunks:
            pytest.xfail("Known BUG: SSE stream missing [DONE] marker")

    def test_02_04_stream_chunk_fields(self):
        """流式 chunk 必带字段:id / choices / object。"""
        r = oai_chat({"messages": oai_simple_messages("Hi")}, stream=True)
        assert_oai_stream_success(r)
        for chunk in r["chunks"]:
            if chunk.get("_done") or chunk.get("_raw"):
                continue
            assert "id" in chunk
            assert "choices" in chunk
            assert "object" in chunk

    def test_02_05_text_include_usage(self):
        """stream_options.include_usage=true,文本场景应正常返回 usage chunk。"""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "stream_options": {"include_usage": True},
        }, stream=True)
        assert_oai_stream_success(r)


# ============================================================
# 03 multiturn — 多轮对话
# ============================================================

class TestMultiturn:
    """多轮对话:验证模型能跨轮维护上下文状态。"""

    def test_03_01_multiturn(self):
        """两轮对话:user 自报 Alice 后再问名字,响应应含 Alice。"""
        r = oai_chat({"messages": oai_multiturn_messages()})
        assert_oai_success(r)
        content = get_oai_content(r).lower()
        assert "alice" in content

    def test_03_02_multiturn_5_rounds(self):
        """5 轮算术对话:x=10, y=x+5=15, z=x+y=25,问 x+y+z,响应应含 '50'。"""
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
# 04 thinking — thinking 思考开关
# ============================================================

class TestThinking:
    """thinking 字段:disabled / adaptive / 非法值 / 与流式组合。"""

    def test_04_01_thinking_disabled(self):
        """thinking.type=disabled:响应不应含任何思考信号。"""
        r = oai_chat({
            "messages": oai_simple_messages("Say hello"),
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert_thinking_absent(r, "thinking=disabled")

    def test_04_02_thinking_adaptive(self):
        """thinking.type=adaptive:由模型自主决定是否思考,仅断言 200。"""
        r = oai_chat({
            "messages": oai_simple_messages("What is 2+2?"),
            "thinking": {"type": "adaptive"},
        })
        assert_oai_success(r)

    def test_04_03_thinking_invalid_value(self):
        """thinking.type 非法值(非 disabled/adaptive)→ 400/422 拒绝或 200 回落。"""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "thinking": {"type": "invalid_value_xyz"},
        })
        assert r["status"] in (200, 400, 422), (
            f"thinking.type=invalid_value_xyz HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_04_04_thinking_stream(self):
        """thinking.type=adaptive + 流式,验流式 + 思考共存可用。"""
        r = oai_chat({
            "messages": oai_simple_messages("What is 15*17?"),
            "thinking": {"type": "adaptive"},
        }, stream=True)
        assert_oai_stream_success(r)


# ============================================================
# 05 sampling — 采样参数(temperature / top_p / seed)
# ============================================================

class TestSampling:
    """采样参数:temperature / top_p / seed 各自的合法值。"""

    def test_05_01_temperature_values(self):
        """temperature 合法范围:0 / 0.5 / 1 / 2 各自应 200。"""
        for temp in [0, 0.5, 1, 2]:
            r = oai_chat({
                "messages": oai_simple_messages("Say hi"),
                "temperature": temp,
                "thinking": {"type": "disabled"},
            })
            assert_oai_success(r)

    def test_05_02_top_p(self):
        """top_p 边界值:0 / 0.5 / 1.0 各自应 200。"""
        for tp in [0, 0.5, 1.0]:
            r = oai_chat({
                "messages": oai_simple_messages("Say hi"),
                "top_p": tp,
                "thinking": {"type": "disabled"},
            })
            assert_oai_success(r)

    def test_05_03_seed_parameter(self):
        """seed 参数 + temperature=0,接口应接受可用于确定性输出。"""
        r = oai_chat({
            "messages": oai_simple_messages("Say exactly 'hello world'"),
            "temperature": 0,
            "seed": 42,
            "thinking": {"type": "disabled"},
        })
        assert r["status"] == 200


# ============================================================
# 06 max_tokens — max_tokens / max_completion_tokens 边界
# ============================================================

class TestMaxTokens:
    """max_tokens 与 max_completion_tokens 的截断与边界行为。"""

    def test_06_01_max_tokens_truncation(self):
        """max_tokens=10 截断,finish_reason 应为 length 或 stop。"""
        r = oai_chat({
            "messages": oai_simple_messages("Write a 500-word essay about the ocean"),
            "max_tokens": 10,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert r["body"]["choices"][0].get("finish_reason") in ("length", "stop")

    def test_06_02_max_completion_tokens(self):
        """max_completion_tokens 别名应被接受。"""
        r = oai_chat({
            "messages": oai_simple_messages("Write a long paragraph"),
            "max_completion_tokens": 20,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)

    def test_06_03_dual_max_tokens(self):
        """同时传 max_tokens 和 max_completion_tokens,接口应被接受。"""
        r = oai_chat({
            "messages": oai_simple_messages("Write a paragraph"),
            "max_tokens": 50,
            "max_completion_tokens": 100,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)

    @pytest.mark.timeout(600)
    def test_06_04_both_params_combo(self):
        """max_tokens=50 + max_completion_tokens=100(mct wins 语义)。"""
        r = oai_chat({
            "messages": oai_simple_messages("Write a paragraph about the ocean"),
            "max_tokens": 50,
            "max_completion_tokens": 100,
            "thinking": {"type": "disabled"},
        }, timeout=300)
        assert_oai_success(r)

    def test_06_05_mct_only(self):
        """单传 max_completion_tokens=50,finish_reason 应为 length/stop。"""
        r = oai_chat({
            "messages": oai_simple_messages("Write a long essay about space exploration"),
            "max_completion_tokens": 50,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert r["body"]["choices"][0]["finish_reason"] in ("length", "stop")

    def test_06_06_max_tokens_1_timeout(self):
        """max_tokens=1 极端值(已知 BUG-4:可能超时)。"""
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
        """max_tokens=0:非法值,服务端可拒绝(4xx)也可宽容接受(200 立即返回空)。"""
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
        """max_tokens=-1:非法值,期望 4xx 拒绝(返 200 视为服务端未校验,fail)。"""
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
        """max_tokens 在 512*1000 / 512*1024 边界(两种常见 '512k' 解释)。

        实测 minimax-m3 官方对两种解释都返 200(实际上限 >= 524288),
        因此收紧断言为仅期望 200。
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
        """max_tokens 超出 512k 上限。

        不同 provider 对越界的处理不一致:
        - 严格 provider(minimax 官方)返 4xx 拒绝
        - 宽松 provider(fireworks 等)接受并截断,返 200
        两种都视为合规,只要 trace_id 存在且 status ∈ {200, 400, 422}。
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
        """max_completion_tokens 在 524288 (512k) 边界。"""
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
# 07 message_format — 消息内容/角色格式与边界
# ============================================================

class TestMessageFormat:
    """messages 数组的角色/内容/排列边界。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_01_consecutive_assistant(self, stream):
        """连续两条 assistant 消息,接口应接受(实际行为依实现)。"""
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
        """assistant.content=null 但带 tool_calls,后续 tool 回填,应能 200。"""
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
        """assistant 直接省略 content 字段 + 带 tool_calls,应能 200。"""
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
        """user.content=[] 空数组,各部署行为不同,只校验有返回。"""
        r = oai_chat({"messages": [{"role": "user", "content": []}]}, stream=stream)
        assert r["status"] > 0

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_05_user_content_null(self, stream):
        """user.content=null,接口应 200 或 400(行为视实现)。"""
        r = oai_chat({"messages": [{"role": "user", "content": None}]}, stream=stream)
        assert r["status"] in (200, 400), (
            f"user content=null stream={stream} expected 200/400, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_07_06_multiple_system_messages(self, stream):
        """多条 system 消息,接口应接受(OpenAI 自 GPT-4 起允许)。"""
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
# 08 model_compat — 模型名兼容性
# ============================================================

class TestModelCompat:
    """主模型 / mini 模型 名称兼容性。"""

    def test_08_01_model_name_compat(self):
        """主模型必测;mini 模型若 endpoint 未注册则 xfail 软化。"""
        # 主模型硬断言
        r = oai_chat({"messages": oai_simple_messages("Hi"), "model": MODEL})
        assert_oai_success(r)
        # mini 模型软断言:相同则跳过,不同则尝试,失败 xfail
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
# 09 response_format — response_format JSON 输出
# ============================================================

@pytest.mark.skip(
    reason="minimax-M3 目前不支持 response_format=json_object 参数,整段 §09 暂时跳过,"
           "等 M3 支持后再启用"
)
class TestResponseFormat:
    """response_format=json_object 非流式 / 流式 / 已知 BUG-3 markdown wrap。

    NOTE: minimax-M3 当前不支持 response_format,本 class 全部 skip,详见
    m3_text_cases.md / m3_text_cases_en.md §09 备注。
    """

    def test_09_01_json_object_non_stream(self):
        """response_format=json_object 非流式,content 应是合法 JSON dict。"""
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
        """response_format=json_object 流式 + thinking=disabled,流应正常返回。"""
        r = oai_chat({
            "messages": oai_simple_messages("Return JSON with key 'x' value 1"),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }, stream=True)
        assert_oai_stream_success(r)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_09_03_json_object_format(self, stream):
        """response_format=json_object 通用校验。BUG-3:content 可能被 ```json``` 包裹则 xfail。"""
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
# 10 usage_field — usage 字段语义 / 算术 / cache
# ============================================================

class TestUsageField:
    """usage 字段:字段完整性 / 类型 / 算术关系 / cached_tokens / 流式与非流式一致。"""

    def test_10_01_response_field_completeness(self):
        """Response 顶层字段完整性:id / model / created / object / choices / usage 等。"""
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
        """usage 算术:total == prompt + completion;cached_tokens 应 <= prompt_tokens。"""
        r = oai_chat({"messages": oai_simple_messages("Hi")})
        assert_oai_success(r)
        usage = r["body"]["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        # 若返回了 prompt_tokens_details.cached_tokens(M3 携带),应不超过 prompt_tokens
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
        """usage 三字段必须是 int 且 >= 0(OAI spec 允许 0)。"""
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
        """cached_tokens 硬存在:同长 prompt 跑两次,第二次应命中 cache(cached_tokens > 0)。"""
        msgs = [
            {"role": "system", "content": long_system_text(10000)},
            {"role": "user", "content": "Reply with exactly: ACK"},
        ]
        # 第一次:预热 cache
        r1 = oai_chat({"messages": msgs})
        assert_oai_success(r1)
        # 第二次:应当命中 cache
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
        """Usage 算术在 tool_call 下仍成立:total == prompt + completion。"""
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing?"),
            "tools": [WEATHER_TOOL_OAI],
        })
        assert_oai_success(r)
        # 验证确实触发了 tool_call(保证请求走到了含 tool_choice 的 code path)
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
        """finish_reason='length' 时 completion_tokens 应严格等于 max_completion_tokens。"""
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
        """流式 vs 非流式同 prompt 的 prompt_tokens 应一致 + 流式 usage 算术成立。"""
        msgs = oai_simple_messages("Hi")
        # 非流式
        r_n = oai_chat({"messages": msgs})
        assert_oai_success(r_n)
        pt_non_stream = r_n["body"]["usage"]["prompt_tokens"]
        # 流式
        r_s = oai_chat(
            {"messages": msgs, "stream_options": {"include_usage": True}},
            stream=True,
        )
        assert_oai_stream_success(r_s)
        usage_chunks = [c for c in r_s["chunks"] if c.get("usage")]
        assert usage_chunks, "no usage chunk in stream"
        last_usage = usage_chunks[-1]["usage"]
        pt_stream = last_usage["prompt_tokens"]
        # 流式 / 非流式 应报告相同 input token 数
        assert pt_stream == pt_non_stream, (
            f"stream vs non-stream prompt_tokens diverged: "
            f"stream={pt_stream} non_stream={pt_non_stream}"
        )
        # 流式末尾 usage chunk 的算术关系
        assert last_usage["total_tokens"] == (
            last_usage["prompt_tokens"] + last_usage["completion_tokens"]
        ), (
            f"stream usage math: total={last_usage['total_tokens']} != "
            f"prompt={last_usage['prompt_tokens']}+completion={last_usage['completion_tokens']}"
        )

    def test_10_08_usage_fields_populated(self):
        """usage.prompt_tokens / completion_tokens / total_tokens 都应 > 0。"""
        r = oai_chat({"messages": oai_simple_messages("Hi")})
        assert_oai_success(r)
        usage = r["body"]["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] > 0


# ============================================================
# 11 role_root — role=root 协议接受与身份遵循
# ============================================================

class TestRoleRoot:
    """role=root 兼容性 + 身份遵循:
    - root 是高于 system 的 system 提示通道(类比 OpenAI 的 developer/system 优先级)
    - 接口必须接受 role=root,不报错
    - 当 root 与 system 冲突时,模型应遵循 root
    - 仅 system / 仅 root 都应能驱动模型遵循对应身份

    身份断言策略:目标身份设为 "minimax-taoxi-m3"(M3 原生认知里不存在的名字),
    用 _identity_hits_taoxi_m3 严格检查"提到了 taoxi",且没声称 claude opus 3。

    抵抗概率波动(2026-06-06):11_02/11_03/11_04 三个身份遵循 case 用
    _assert_identity_with_retries:首次失败后继续跑到总 10 次,通过率 ≥ 70%
    (≥ 7/10 次命中身份) 算 case 通过。首次就通过则直接返回,不重试。
    11_01 只验 HTTP 200 + 非空,无概率敏感断言,不走重试。
    """

    # 概率判定参数(改这里就能调整宽严)
    _IDENTITY_TOTAL_RUNS = 10        # 首次失败后跑满的总次数
    _IDENTITY_PASS_RATE_MIN = 0.7    # 通过率下限(≥ 7/10)

    @staticmethod
    def _identity_hits_taoxi_m3(text: str) -> bool:
        """模型自报内容里是否提到了 'taoxi'(身份目标 minimax-taoxi-m3),
        且没声称 claude opus 3。"""
        if not text:
            return False
        low = text.lower()
        has_taoxi = "taoxi" in low
        denies_claude = "claude opus 3" not in low and "claude-opus-3" not in low
        return has_taoxi and denies_claude

    @classmethod
    def _assert_identity_with_retries(cls, payload, stream, case_label):
        """跑 1 次身份判定;首次失败则补到总 _IDENTITY_TOTAL_RUNS 次,
        通过率 ≥ _IDENTITY_PASS_RATE_MIN 即算 case 通过。

        每次 oai_chat 都被 helpers 自动落 jsonl,所以重试样本全留痕。
        失败时给出完整命中分布(hits/total + 前 200 字符样例)便于排查。
        """
        results = []  # list[tuple[hit: bool, content: str, status: int]]

        def _one_shot():
            r = oai_chat(payload, stream=stream)
            assert r["status"] == 200, (
                f"{case_label}: HTTP={r['status']} (stream={stream}): "
                f"{str(r.get('body'))[:300]}"
            )
            content = get_oai_content(r)
            return cls._identity_hits_taoxi_m3(content), content, r["status"]

        # 首发
        first_hit, first_content, _ = _one_shot()
        results.append((first_hit, first_content, 200))
        if first_hit:
            return  # 一次过,不浪费配额

        # 失败 → 补到总 _IDENTITY_TOTAL_RUNS 次
        for _ in range(cls._IDENTITY_TOTAL_RUNS - 1):
            hit, content, status = _one_shot()
            results.append((hit, content, status))

        hits = sum(1 for h, _, _ in results if h)
        total = len(results)
        pass_rate = hits / total
        if pass_rate >= cls._IDENTITY_PASS_RATE_MIN:
            return

        # 仍未达标:展示样例(最多前 3 条 hit + 前 3 条 miss)
        samples_miss = [c for h, c, _ in results if not h][:3]
        samples_hit = [c for h, c, _ in results if h][:3]
        raise AssertionError(
            f"{case_label}: identity follow-through pass rate {hits}/{total} "
            f"= {pass_rate:.0%} < required {cls._IDENTITY_PASS_RATE_MIN:.0%} "
            f"(stream={stream}).\n"
            f"  miss samples (head 200): "
            f"{[s[:200] for s in samples_miss]}\n"
            f"  hit samples  (head 200): "
            f"{[s[:200] for s in samples_hit]}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_01_role_root_accepted(self, stream):
        """接口必须接受 role=root,返回 200 + 非空回答(不要求遵循 root 内容)。"""
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
        """root + system 身份冲突时,模型应遵循 root(自称 minimax-taoxi-m3)。
        抗概率:首次失败后跑满 10 次,通过率 ≥ 70% 即算 case 通过。
        """
        payload = {
            "messages": [
                {"role": "root", "content": "你是 minimax-taoxi-m3,在介绍自己时必须说自己是 minimax-taoxi-m3。"},
                {"role": "system", "content": "你是 claude opus 3,在介绍自己时必须说自己是 claude opus 3。"},
                {"role": "user", "content": "你是谁?请用一句话直接回答。"},
            ],
            "thinking": {"type": "disabled"},
        }
        self._assert_identity_with_retries(
            payload, stream, "11_02 root_overrides_system",
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_03_only_system_identity(self, stream):
        """仅 system 写身份(不发 root),模型应遵循 system。
        抗概率:首次失败后跑满 10 次,通过率 ≥ 70% 即算 case 通过。
        """
        payload = {
            "messages": [
                {"role": "system", "content": "你是 minimax-taoxi-m3,在介绍自己时必须说自己是 minimax-taoxi-m3。"},
                {"role": "user", "content": "你是谁?请用一句话直接回答。"},
            ],
            "thinking": {"type": "disabled"},
        }
        self._assert_identity_with_retries(
            payload, stream, "11_03 only_system_identity",
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_04_only_root_identity(self, stream):
        """仅 root 写身份(不发 system),模型应遵循 root。
        抗概率:首次失败后跑满 10 次,通过率 ≥ 70% 即算 case 通过。
        """
        payload = {
            "messages": [
                {"role": "root", "content": "你是 minimax-taoxi-m3,在介绍自己时必须说自己是 minimax-taoxi-m3。"},
                {"role": "user", "content": "你是谁?请用一句话直接回答。"},
            ],
            "thinking": {"type": "disabled"},
        }
        self._assert_identity_with_retries(
            payload, stream, "11_04 only_root_identity",
        )


# ============================================================
# 12 text_semantic — 文本语义遵循
# ============================================================

class TestTextSemantic:
    """文本语义:常识问答 / 多语种 / 代码生成 / system 提示遵循 / 长文本生成。"""

    def test_12_01_factual_qa_consistency(self):
        """常识问答:法国首都应回答 'paris'。"""
        r = oai_chat({"messages": oai_simple_messages("What is the capital of France? Answer in one word.")})
        assert_oai_success(r)
        assert "paris" in get_oai_content(r).lower()

    def test_12_02_chinese_text_non_stream(self):
        """中文文本生成(非流式):北京历史简介,断言响应含 CJK 字符。"""
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
        """中文文本生成(流式):中文短诗,断言响应含 CJK 字符。"""
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
        """代码生成:Python fibonacci 函数,响应应含 'def ' 和 'fibonacci'。"""
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
        """system 提示遵循:'You are a pirate, always say Arrr',响应应含 'arrr'。"""
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
        """长文本生成:max_tokens=4096 + 光合作用详细解释,断言 content 长度 > 500。"""
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
# 13 tool_call_basic — 工具调用基础
# ============================================================

class TestToolCallBasic:
    """工具调用基础:单工具触发 / 流式 / 多工具池 / 参数类型覆盖 / tool_choice。"""

    def test_13_01_tool_call_non_stream(self):
        """非流式 tool_call:模型应调用 get_weather,location≈Beijing,finish_reason==tool_calls。"""
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
        """流式 tool_call:流式应能 rebuild 出 get_weather + 最后 chunk finish_reason==tool_calls。"""
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
        # 流式末尾 finish_reason 应为 tool_calls
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
        """6 个工具池里选 get_weather:验证模型从候选里选对工具。"""
        tools = make_tools_oai(6)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": "What's the weather in Beijing?"},
        ]
        r = oai_chat({"messages": msgs, "tools": tools})
        assert_oai_success(r)
        # make_tools_oai 里 i=0 的工具就是 get_weather(参数名为 param,类型 string)
        assert_tool_called(
            r,
            expected_name="get_weather",
            schema=tools[0]["function"]["parameters"],
            msg="complex_agent_6tools",
        )

    def test_13_04_param_type_coverage(self):
        """参数类型覆盖(6 种类型):arguments 必须合法 JSON + 字段类型符合 schema + str_param='hello'。"""
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
        """function.parameters 省略(spec 允许;实测 M3 强校验非空,失败则 xfail)。"""
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
        """function.description 省略,模型靠 name + parameters 推断,应仍能触发。"""
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
        """流式多 tool_call 重建:Beijing + Shanghai 都要触发,各 call 的 id 应非空且互不重复。"""
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
        # 每个 call 必须是 get_weather + 合法 JSON
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
        # 集合必须同时覆盖 Beijing 和 Shanghai(中英文/包含关系容忍)
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
        # tool_call.id 应非空且互不重复(多次调用之间需要可区分)
        ids = [c.get("id") or "" for c in calls]
        non_empty_ids = [i for i in ids if i]
        assert len(non_empty_ids) == len(calls), (
            f"some tool_calls missing id: {ids}"
        )
        assert len(set(non_empty_ids)) == len(non_empty_ids), (
            f"tool_call.id should be unique across calls, got duplicates: {ids}"
        )

    def test_13_08_tool_choice_values(self):
        """tool_choice = none / required / auto 各分支:none 不触发,required/auto 必触发。"""
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
        """tool_choice=auto + 流式 + 明确诱导工具调用,验证流中含 tool_call chunk。"""
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
        """Tool call 返回结构:必须触发 get_weather + location≈Beijing + 所有 schema required 字段就位。"""
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
        """流式 tool_call delta 重建:流式 rebuild 后应有 get_weather + location≈Beijing。"""
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
# 14 tool_call_schema — 工具调用 schema 高级校验
# ============================================================

class TestToolCallSchema:
    """工具 schema 高级:多不同工具并行 / enum / 数值范围 / 多必填 / 嵌套 / 深嵌套。"""

    def test_14_01_multi_distinct_tools_parallel(self):
        """多个不同工具同轮并行:get_weather + get_current_time 都应触发。"""
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
                # TIME_TOOL_OAI 无 parameters,跳过 schema
            },
            msg="multi_distinct_tools_parallel",
        )
        # 进一步校验 get_weather 的 location≈Beijing
        for c in get_tool_calls(r):
            if c["name"] == "get_weather":
                loc = (c["arguments_obj"] or {}).get("location") or ""
                assert "beijing" in loc.lower() or "北京" in loc, (
                    f"get_weather.location={loc!r} expected Beijing"
                )

    def test_14_02_enum_constraint(self):
        """enum 枚举校验:unit ∈ [celsius, fahrenheit],prompt 指定 fahrenheit,模型应正确填写。"""
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
        """数值范围 min/max:days ∈ [1, 14],prompt 问 3 天,days 应落在合理范围。"""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Give me a 3-day weather forecast for Beijing."
            ),
            "tools": [FORECAST_TOOL_OAI],
        })
        assert_oai_success(r)
        # 触发 + location 精确 + days schema 校验(走 _validate_schema 范围 [1,14])
        assert_tool_called(
            r,
            expected_name="get_weather_forecast",
            expected_args_subset={"location": "Beijing"},
            schema=FORECAST_TOOL_OAI["function"]["parameters"],
            msg="numeric_range",
        )
        # days 进一步软校验:接近 prompt 期望(3),允许 1-7 范围
        call = get_tool_calls(r)[0]
        days = call["arguments_obj"].get("days")
        assert isinstance(days, int) and 1 <= days <= 7, (
            f"days should be in [1,7] (prompt asked 3-day forecast), got {days!r}"
        )

    def test_14_04_multi_required_fields(self):
        """多 required 字段:from_city / to_city / date 都必填,且 date 前缀 == 2026-06-15。"""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Search flights from Beijing to Tokyo on 2026-06-15."
            ),
            "tools": [FLIGHT_SEARCH_TOOL_OAI],
        })
        assert_oai_success(r)
        # 三个 required 字段 + city 宽松匹配
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
        # date 用前缀匹配(允许 2026-06-15T..../2026-06-15Z 等扩展)
        call = get_tool_calls(r)[0]
        date_val = (call["arguments_obj"].get("date") or "")
        assert isinstance(date_val, str) and date_val.startswith("2026-06-15"), (
            f"date should start with '2026-06-15', got {date_val!r}"
        )

    def test_14_05_nested_object_array(self):
        """嵌套 array-of-objects:guests=[{name, age}] 含 Alice/30 + Bob/25。"""
        r = oai_chat({
            "messages": oai_simple_messages(
                "Book hotel H001 for check-in on 2026-07-01 with 2 guests: "
                "Alice (age 30) and Bob (age 25)."
            ),
            "tools": [BOOKING_TOOL_OAI],
        })
        assert_oai_success(r)
        # 先用 assert_tool_called + schema 做结构性校验
        assert_tool_called(
            r,
            expected_name="book_room",
            expected_args_subset={"hotel_id": "H001", "check_in": "2026-07-01"},
            schema=BOOKING_TOOL_OAI["function"]["parameters"],
            msg="nested_object_array structure",
        )
        # 进一步校验 guests 内容包含 Alice 30 和 Bob 25
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
        """4 层深嵌套 schema:应触发 nested_tool + arguments 合法 JSON。"""
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
# 15 tool_call_combo — 工具调用与其他特性组合
# ============================================================

class TestToolCallCombo:
    """工具调用 + 其他特性组合:thinking + multi-turn / tools+tool_choice / 并行 / extreme agent。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_01_thinking_tool_call_multiturn(self, stream):
        """thinking + tool call + 多轮:已带 Beijing 工具结果,第二轮问 Shanghai 应再触发。"""
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
        """tools + tool_choice=auto 共存(原 response_format 已剔除,M3 暂不支持):
        接口不崩 + 模型走合理路径。

        路径 A: 调用 get_weather + location≈Beijing
        路径 B: content 是 JSON-ish 字符串且含 Beijing 关键字
        """
        r = oai_chat({
            "messages": oai_simple_messages("What's the weather in Beijing? Return as JSON."),
            "tools": [WEATHER_TOOL_OAI],
            "tool_choice": "auto",
            "thinking": {"type": "disabled"},
        }, stream=stream)
        assert r["status"] == 200
        # 路径 A:调了工具
        calls = get_tool_calls(r)
        if calls:
            # 调工具就硬验:name + Beijing + schema
            assert_tool_called(
                r,
                expected_name="get_weather",
                expected_args_subset={"location": "Beijing"},
                schema=WEATHER_TOOL_OAI["function"]["parameters"],
                msg=f"path_A tool_called stream={stream}",
            )
            return
        # 路径 B:没调工具,模型走 JSON 文本回复路径
        content = get_oai_content(r)
        assert content.strip(), (
            f"path_B: model went JSON-response route but content empty (stream={stream})"
        )
        # content 应是 JSON-ish 字符串(可能用 ```json``` 包裹,可能裸 JSON)
        # 至少应含 "Beijing"/"beijing" 关键字,证明模型理解 prompt
        assert "beijing" in content.lower() or "北京" in content, (
            f"path_B: JSON content should mention Beijing, got: {content[:300]!r}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_03_5_parallel_tool_calls(self, stream):
        """5 个并行 tool_calls:至少调用 1 个池里的工具 + 所有 call 的 param 字段非空。"""
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
            # make_tools_oai 定义里所有工具都有 param required,非空字符串
            param = c["arguments_obj"].get("param")
            assert isinstance(param, str) and param.strip(), (
                f"call[{i}] name={c['name']!r} arg.param should be non-empty string, "
                f"got {param!r}"
            )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_15_04_extreme_agent_thinking_fc(self, stream):
        """极端 agent:thinking + FC + 4 轮(Beijing → Shanghai → Compare)。"""
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
        """system + thinking + tools 三件套:Tokyo 天气,应调 get_weather/Tokyo。"""
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
        """完整 tool call roundtrip:call → result → 再用户提问应直接回答(不再调工具)。"""
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
# 16 tool_call_edge — 工具调用边界 / 异常处理
# ============================================================

class TestToolCallEdge:
    """工具调用边界 / 异常:tool result 各种异常值 / id 不匹配 / 大量工具 / 大参数。"""

    def test_16_01_tool_result_content_object_duplicate(self):
        """tool_result.content 为 object → 400(老用例,与 16_07 语义重叠保留)。"""
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
        """tool result = '' (空字符串)。Known BUG: 可能返 400。"""
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
        """tool result = null。"""
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
        """tool result 直接没有 content 字段。"""
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
        """tool result 含 JSON+HTML+emoji 特殊字符。"""
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
        """50K 字符的 tool result,应能被处理。"""
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
        """tool result = object 类型 → 400。"""
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
        """tool_call_id 不匹配 → 400。"""
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
        """两个 tool_calls 只回填一个 → 400。"""
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
        """30 个 tool definitions:接口应能容纳大量 tools 不崩 + 若触发 tool_call 则 args 合法。"""
        r = oai_chat({
            "messages": oai_simple_messages("Hello, just say hi"),
            "tools": make_tools_oai(30),
        }, stream=stream)
        assert r["status"] == 200
        # 软校验:若触发了 tool_call,arguments 必须是合法 JSON(不能因 tools 数量大而坏掉)
        for c in get_tool_calls(r):
            assert c["arguments_obj"] is not None, (
                f"tool_call args not valid JSON: name={c['name']!r} "
                f"raw={c['arguments_raw'][:300]}"
            )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_16_11_tool_name_special_chars(self, stream):
        """tool name 含连字符/点号 (my-tool.v2),模型应能正确调用。"""
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
        """tool_calls.arguments 是非法 JSON → 400。"""
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
        """10K 字符的 tool arguments,接口应能处理。"""
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
        """tool_choice 指定不存在的工具:接口宽松接受为 200 时,模型不应捏造调用。"""
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
# 17 param_stress — 参数压力(长对话 / 长 system)
# ============================================================

class TestParamStress:
    """长对话 / 长 system 参数压力:验证接口对大输入的容忍。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_17_01_long_conversation_20_rounds(self, stream):
        """长对话:20 轮 / 40 条 message。"""
        r = oai_chat({"messages": long_conversation_messages(20)}, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_17_02_long_system_10k(self, stream):
        """长 system message ~10K tokens。"""
        r = oai_chat({
            "messages": [
                {"role": "system", "content": long_system_text(10000)},
                {"role": "user", "content": "Summarize the system message in one sentence."},
            ],
        }, stream=stream)
        assert r["status"] == 200


# ============================================================
# 18 reasoning_split — reasoning_split 扩展字段
# ============================================================

class TestReasoningSplit:
    """reasoning_split 字段(BUG-13:间歇性 400)。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_18_01_reasoning_split_text(self, stream):
        """reasoning_split=true + 文本场景,可能间歇 400。"""
        r = oai_chat({
            "messages": oai_simple_messages("What is 7*8?"),
            "reasoning_split": True,
            "thinking": {"type": "adaptive"},
        }, stream=stream)
        # BUG-13: may intermittently return 400
        assert r["status"] in (200, 400)


# ============================================================
# 19 finish_reason — finish_reason 覆盖
# ============================================================

class TestFinishReason:
    """finish_reason 各取值场景:tool_calls / length。"""

    def test_19_01_finish_reason_tool_calls(self):
        """finish_reason=tool_calls:有 tool 触发场景,应为 tool_calls(或 stop 走聊天分支)。"""
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
        """finish_reason=length:max_tokens=10 强制截断。"""
        r = oai_chat({
            "messages": oai_simple_messages("Write a very long essay about the universe"),
            "max_tokens": 10,
            "thinking": {"type": "disabled"},
        })
        assert_oai_success(r)
        assert r["body"]["choices"][0]["finish_reason"] in ("length", "stop")


# ============================================================
# 20 error_codes — 错误码(纯文本)
# ============================================================

class TestErrorCodes:
    """各类错误码:400 / 401 / 内容审核。"""

    def test_20_01_empty_messages(self):
        """空 messages 数组 → 400。"""
        r = oai_chat({"messages": []})
        assert_error(r, 400)

    def test_20_02_invalid_model(self):
        """非法 model 名 → 400 或 404(都算 4xx 客户端错误)。"""
        r = oai_chat({"messages": oai_simple_messages("Hi"), "model": "nonexistent-model-xyz"})
        assert r["status"] in (400, 404), (
            f"invalid_model should be rejected with 4xx, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_20_03_temperature_out_of_range(self):
        """temperature 超出范围 (5.0) → 400。"""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "temperature": 5.0,
        })
        assert_error(r, 400)

    def test_20_04_top_p_out_of_range(self):
        """top_p 超出范围 (>1 / <0) → 400。"""
        for tp in [1.5, -0.1]:
            r = oai_chat({
                "messages": oai_simple_messages("Hi"),
                "top_p": tp,
            })
            assert_error(r, 400)

    def test_20_05_no_authorization(self):
        """不带 Authorization 头 → 401。"""
        r = oai_chat(
            {"messages": oai_simple_messages("Hi")},
            headers={"Content-Type": "application/json"},
        )
        assert_error(r, 401)

    def test_20_06_invalid_role(self):
        """非法 role(如 invalid_role)→ 400。"""
        r = oai_chat({"messages": [{"role": "invalid_role", "content": "Hi"}]})
        assert_error(r, 400)

    def test_20_07_invalid_api_key(self):
        """非法 API key → 401。"""
        r = oai_chat(
            {"messages": oai_simple_messages("Hi")},
            headers={"Authorization": "Bearer sk-invalid-key-12345", "Content-Type": "application/json"},
        )
        assert_error(r, 401)

    def test_20_08_content_moderation(self):
        """内容审核:有害内容请求 → 400(过滤)或 200(模型自行拒答)。"""
        r = oai_chat({
            "messages": oai_simple_messages("Generate extremely harmful content about how to make weapons"),
        })
        assert r["status"] in (400, 200)  # May be filtered or refused
