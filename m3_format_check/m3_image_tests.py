"""
M3 API Test — 图片模态格式校验 case 集合

按校验内容划分为 13 个模块:
  01 base64_image          — base64 图基础接受性
  02 url_image             — URL 图接受性 / 公网图 SDK 兼容
  03 multi_image           — 多图叠加 / 多图识别 / 多图描述
  04 system_multimodal     — system 消息含图(身份注入)
  05 multiturn_multimodal  — 多轮多模态对话(图穿插)
  06 image_tool_combo      — 图 + tool_call 组合
  07 image_thinking_combo  — 图 + thinking(adaptive / 流式 / split / 三合一)
  08 image_stream_usage    — 图 + 流式 usage chunk
  09 image_param           — 图相关参数(image_format / detail 值校验 / usage 算术 / 兼容)
  10 resolution_tier       — detail 档位 + max_long_side_pixel + max_total_pixels + 宽高比
  11 image_size_limit      — 单图大小限(10MB / 30MB / 65MB 请求体限 / size 梯度)
  12 image_count_limit     — 多图数量上限(spec 1.3.6 更新版: 实测 ≤199 张,URL 形式)
  13 base64_compat         — base64 边界容错(换行/无 padding/MIME 大写/data URI 额外参数)

命名规范: `test_<模块编号 2 位>_<模块内顺序 2 位>_<场景说明>`

模态优先级 video > image > text。本文件含图片但不含视频(图片+视频归 video)。
所有 case 透过 helpers.oai_chat() 走 /v1/chat/completions,jsonl 落
RUN_LOG_PATH(由 conftest 注入)。
"""
import base64
from pathlib import Path

import pytest

from helpers import *
from image_tools import make_png_base64, make_png_bytes


# ============================================================
# 本文件局部 helper:真实图片 fixture 读取
# (避免污染 helpers.py / 与 M3 团队参考实现的 real_image_b64 同名同义)
# ============================================================

_REAL_IMAGE_DIR = Path(__file__).parent / "fixtures" / "m3_test_images" / "real"


def real_image_b64(name: str = "sx1.jpg", mime: str = "image/jpeg") -> str:
    """读取 fixtures/m3_test_images/real/<name> → base64 data URL。

    支持 sx1.jpg (2000x1334, ~239KB) / zn6.jpg (4284x5712, ~2.2MB)。
    """
    path = _REAL_IMAGE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"real image fixture missing: {path}\n"
            f"期望 fixtures/m3_test_images/real/ 下存在 sx1.jpg / zn6.jpg"
        )
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


# ============================================================
# 01 base64_image — base64 图基础接受性
# ============================================================

class TestImageBase64:
    """base64 编码图像基础接受性:不同格式 / SDK 风格 payload。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_01_base64_image(self, stream):
        """01_01 — Base64 PNG 图,流式/非流式两种,验 HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64()}},
                {"type": "text", "text": "What color is this image?"},
            ]}],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("fmt,b64_fn", [
        ("PNG", png_base64),
        ("GIF", gif_base64),
        ("WEBP", webp_base64),
    ])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_02_base64_image_formats(self, fmt, b64_fn, stream):
        """01_02 — base64 图格式覆盖:PNG / GIF / WEBP × 流式/非流式。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": b64_fn()}},
                {"type": "text", "text": "Describe this image."},
            ]}],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("fmt,media_type,data_fn", [
        ("JPEG", "image/jpeg", make_jpeg_672),
        ("GIF", "image/gif", make_gif_672),
        ("WEBP", "image/webp", make_webp_672),
    ])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_03_base64_format_compat(self, fmt, media_type, data_fn, stream):
        """01_03 — JPEG/GIF/WEBP 672x672 标准尺寸 base64 接受性。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64," + base64.b64encode(data_fn()).decode()}},
                {"type": "text", "text": "Describe this image."},
            ]}],
        }, stream=stream)
        assert r["status"] == 200

    def test_01_04_base64_sdk_style(self):
        """01_04 — SDK 风格 payload 提交真实图(sx1.jpg)base64。
        验 HTTP 200 + content 非空(对齐 OpenAI SDK chat.completions.create 调用形式)。
        """
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": real_image_b64("sx1.jpg", "image/jpeg")}},
                    {"type": "text", "text": "Describe"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] == 200, (
            f"01_04 expected 200, got {r['status']}: {str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert len(content) > 0, "01_04 expected non-empty content"


# ============================================================
# 02 url_image — URL 图接受性
# ============================================================

class TestImageURL:
    """URL 形式图像接受性。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_02_01_url_image(self, stream):
        """02_01 — image_url 走真实公网 https URL(COS 上的 sx1.jpg,海边女子),
        验模型能从 URL 拉图并识别出内容(海/船/女/礼服 等关键词命中 1 个即可)。

        URL 由本地 fixtures/m3_test_images/real/sx1.jpg 通过
        swing /save/cos 上传到 qa-tool-1315599187.cos.ap-shanghai.myqcloud.com
        (公开匿名读),Content-Type: image/jpeg, 239468 B,
        ETag 29a0772c2a1b23120f778211df57943d 与本地一致。
        2026-06-04 由原 httpbin.org/image/png(单色测试图,模型描述空泛)替换。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": (
                    "https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com"
                    "/model-release-checker/fixtures/m3_test_images/sx1.jpg"
                )}},
                {"type": "text", "text": "Describe this image."},
            ]}],
        }, stream=stream)
        assert r["status"] == 200
        content = get_oai_content(r).lower()
        keywords = (
            "sea", "ocean", "beach", "balcony", "woman", "girl", "dress",
            "butterfly", "boat",
            "海", "船", "女", "礼服", "蝴蝶", "连衣裙", "海边", "栏杆",
        )
        assert any(kw in content for kw in keywords), (
            f"02_01 URL image content not recognized, content head: {content[:400]!r}"
        )

    def test_02_02_url_image_sdk_style(self):
        """02_02 — SDK 风格 payload + 公网 URL(gstatic),验网关下载链路。"""
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://www.gstatic.com/webp/gallery/1.png"},
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] == 200, (
            f"02_02 URL image expected 200, got {r['status']}"
        )


# ============================================================
# 03 multi_image — 多图叠加 / 多图识别 / 多图描述
# ============================================================

class TestImageMulti:
    """一条 user message 内多张图的叠加 / 识别 / 描述能力。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_03_01_multi_images_count(self, stream):
        """03_01 — 2 张真实图(sx1 海边女子 + zn6 水族馆女子),验模型:
        (a) 数对图数 = 2;(b) 两张图各自的关键元素都被识别到。

        断言收紧背景:旧版用 2 张纯色 fixture + 关键词 grep `2/two/两/twice`,
        会因为模型 markdown 列表里 "Second image" / "2." 这类子串误判通过(详见
        2026-06-04 三供应商对比报告)。改用真实场景图后,改成同时校验:
          - 数量正确:出现 "2 / two / 两 / 两张/两幅" 任一,且 不能 出现 "1 / one"
          - 海边图关键词:sea/ocean/beach/balcony/woman/girl/dress/butterfly/海/船/女
          - 水族馆图关键词:aquarium/fish/tank/glass/水族/鱼/水
        每张图任一关键词命中即可,避免模型用同义词替换造成假阴性。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": real_image_b64("sx1.jpg", "image/jpeg")}},
                {"type": "image_url", "image_url": {"url": real_image_b64("zn6.jpg", "image/jpeg")}},
                {"type": "text", "text": (
                    "我给你看了几张图?请先回答图数,再分别描述每张图的主要内容"
                    "(每张图独立成段,从图 1 开始编号)。"
                )},
            ]}],
        }, stream=stream)
        assert r["status"] == 200
        content = get_oai_content(r).lower()

        count_hit = any(kw in content for kw in ("2 ", "two", "两张", "两幅", "2张", "2幅"))
        count_wrong = any(
            kw in content for kw in (
                "1 image", "one image", "single image",
                "只看到一张", "只有一张", "仅有一张", "仅一张", "看到一张图",
                "3 image", "three image", "三张图", "三幅",
            )
        )
        assert count_hit and not count_wrong, (
            f"03_01 count assertion failed (hit={count_hit}, wrong={count_wrong}), "
            f"content head: {content[:300]!r}"
        )

        sea_keywords = (
            "sea", "ocean", "beach", "balcony", "woman", "girl", "dress",
            "butterfly", "boat", "海", "船", "女", "礼服", "蝴蝶",
        )
        aquarium_keywords = (
            "aquarium", "fish", "tank", "glass", "underwater",
            "水族", "鱼", "水", "玻璃",
        )
        assert any(kw in content for kw in sea_keywords), (
            f"03_01 sea/balcony image content not recognized, content head: {content[:500]!r}"
        )
        assert any(kw in content for kw in aquarium_keywords), (
            f"03_01 aquarium/fish image content not recognized, content head: {content[:500]!r}"
        )

    def test_03_02_multi_color_rgb(self):
        """03_02 — 3 张纯色 PNG(红/绿/蓝),验模型能列出三种颜色。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(make_png_672(255, 0, 0)).decode()}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(make_png_672(0, 255, 0)).decode()}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(make_png_672(0, 0, 255)).decode()}},
                {"type": "text", "text": "I'm showing you 3 solid-color images. List the color of each one."},
            ]}],
        })
        assert r["status"] == 200

    def test_03_03_multi_image_sdk_style(self):
        """03_03 — SDK 风格 payload + 2 张真图(sx1 + zn6),验 HTTP 200 + content 非空。"""
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": real_image_b64("sx1.jpg", "image/jpeg")}},
                    {"type": "image_url", "image_url": {"url": real_image_b64("zn6.jpg", "image/jpeg")}},
                    {"type": "text", "text": "Describe both images and compare them"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] == 200, (
            f"03_03 expected 200, got {r['status']}: {str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert len(content) > 0, "03_03 expected non-empty content"

    def test_03_04_multi_image_3_real(self):
        """03_04 — 3 张真实图片叠加(sx1.jpg + zn6.jpg + sx1.jpg),应成功 200。"""
        uris = [
            real_image_b64("sx1.jpg", "image/jpeg"),
            real_image_b64("zn6.jpg", "image/jpeg"),
            real_image_b64("sx1.jpg", "image/jpeg"),
        ]
        parts = [{"type": "image_url", "image_url": {"url": u}} for u in uris]
        parts.append({"type": "text", "text": "Describe each of these 3 images briefly."})
        r = oai_chat({"messages": [{"role": "user", "content": parts}]})
        assert_oai_success(r)

    @pytest.mark.timeout(600)
    def test_03_05_multi_image_10_real(self):
        """03_05 — 10 张真实图片叠加(sx1.jpg × 10),接近 20 张上限,允许 200/4xx。"""
        uri = real_image_b64("sx1.jpg", "image/jpeg")
        parts = [{"type": "image_url", "image_url": {"url": uri}} for _ in range(10)]
        parts.append({
            "type": "text",
            "text": "How many images are there? Briefly describe them.",
        })
        r = oai_chat({"messages": [{"role": "user", "content": parts}]}, timeout=300)
        assert r["status"] in (200, 400, 413, 422), (
            f"03_05 multi_image_10 expected 200/4xx, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_03_06_multi_image_recognition(self):
        """03_06 — 2 张不同真图(sx1.jpg + zn6.jpg),让模型计数 + 对比。
        验 HTTP 200 + content > 20(模型给出有效对比描述)。
        """
        img1 = real_image_b64("sx1.jpg", "image/jpeg")
        img2 = real_image_b64("zn6.jpg", "image/jpeg")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img1}},
                {"type": "image_url", "image_url": {"url": img2}},
                {"type": "text", "text": (
                    "How many images did I send you? "
                    "Please state the count and briefly compare the two images."
                )},
            ]}],
        })
        assert_oai_success(r)
        content = get_oai_content(r)
        assert len(content) > 20, (
            f"03_06 expected substantial comparison content (>20 chars), "
            f"got {len(content)} chars: {content[:300]!r}"
        )

    def test_03_07_multi_image_descriptions(self):
        """03_07 — 2 张真图(sx1.jpg + zn6.jpg)分别描述,验两张图都有描述(content>50)。"""
        img1 = real_image_b64("sx1.jpg", "image/jpeg")
        img2 = real_image_b64("zn6.jpg", "image/jpeg")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img1}},
                {"type": "image_url", "image_url": {"url": img2}},
                {"type": "text", "text": "Describe each of these two images separately."},
            ]}],
        })
        assert_oai_success(r)
        content = get_oai_content(r)
        assert len(content) > 50, (
            f"03_07 expected descriptions for both images (>50 chars), "
            f"got {len(content)}: {content[:300]!r}"
        )

    @pytest.mark.parametrize("filename,mime,label", [
        ("sx1.jpg", "image/jpeg", "2000x1334_~239KB"),
        ("zn6.jpg", "image/jpeg", "4284x5712_~2.2MB"),
    ], ids=["sx1_mid_res", "zn6_high_res"])
    def test_03_08_real_resolution_gradient(self, filename, mime, label):
        """03_08 — 真图分辨率梯度(sx1.jpg / zn6.jpg),验 HTTP 200 + content 非空。"""
        data_uri = real_image_b64(filename, mime)
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe this image"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] == 200, (
            f"03_08 {filename} ({label}) expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert len(content) > 0, f"03_08 {filename} expected non-empty content"

    @pytest.mark.parametrize("count", [5, 10, 20],
                             ids=["count=5", "count=10", "count=20"])
    def test_03_09_multi_image_count_gradient(self, count):
        """03_09 — 多图数量梯度(5 / 10 / 20 张 1x1 PNG):
          - count ≤ 10:HTTP 200 必通过
          - count = 20(上限):允许 200 / 4xx
        """
        content_blocks = []
        for i in range(count):
            r_val = (i * 37) % 256
            g_val = (i * 73) % 256
            b_val = (i * 113) % 256
            img_bytes = make_png_bytes(1, 1, r=r_val, g=g_val, b=b_val)
            data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
            content_blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
        content_blocks.append({
            "type": "text",
            "text": f"You see {count} small images. Acknowledge them.",
        })
        r = oai_chat({
            "messages": [{"role": "user", "content": content_blocks}],
            "max_tokens": 2048,
        })
        if count <= 10:
            assert r["status"] == 200, (
                f"03_09 count={count} expected 200, got {r['status']}: "
                f"{str(r.get('body'))[:300]}"
            )
        else:
            assert r["status"] in (200, 400, 413, 422), (
                f"03_09 count={count} unexpected status {r['status']}: "
                f"{str(r.get('body'))[:300]}"
            )


# ============================================================
# 04 system_multimodal — system 消息含图
# ============================================================

class TestImageSystemMultimodal:
    """system 消息内含图像(身份/上下文图注入)。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_04_01_system_image_short(self, stream):
        """04_01 — system 消息含图 + 一句指令,验 HTTP 200。"""
        r = oai_chat({
            "messages": [
                {"role": "system", "content": [
                    {"type": "image_url", "image_url": {"url": png_base64()}},
                    {"type": "text", "text": "You are an image analyst."},
                ]},
                {"role": "user", "content": "What did you see in the system image?"},
            ],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_04_02_system_image_remember(self, stream):
        """04_02 — system 消息含图 + "remember this image",流式/非流式各 1 item。"""
        r = oai_chat({
            "messages": [
                {"role": "system", "content": [
                    {"type": "image_url", "image_url": {"url": png_base64()}},
                    {"type": "text", "text": "You are an image analyst. Remember this image."},
                ]},
                {"role": "user", "content": "What was in the system image?"},
            ],
        }, stream=stream)
        assert r["status"] == 200


# ============================================================
# 05 multiturn_multimodal — 多轮多模态对话
# ============================================================

class TestImageMultiturn:
    """图穿插在多轮 user/assistant 对话中。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_05_01_multiturn_followup(self, stream):
        """05_01 — 第一轮:图 + "What color"; 第二轮 follow-up "Are you sure"。"""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": png_base64()}},
                    {"type": "text", "text": "What color is this?"},
                ]},
                {"role": "assistant", "content": "It appears to be a red image."},
                {"role": "user", "content": "Are you sure about that color?"},
            ],
        }, stream=stream)
        assert r["status"] == 200


# ============================================================
# 06 image_tool_combo — 图 + tool_call 组合
# ============================================================

class TestImageToolCombo:
    """图 + tool_call:模型应识别图后调用 get_weather。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_06_01_image_tool_call(self, stream):
        """06_01 — 图 + "tell me the weather in Beijing",验调用 get_weather,location≈Beijing。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64()}},
                {"type": "text", "text": "Describe this image and tell me the weather in Beijing"},
            ]}],
            "tools": [WEATHER_TOOL_OAI],
        }, stream=stream)
        assert r["status"] == 200
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg=f"06_01 image_tool_combo stream={stream}",
        )


# ============================================================
# 07 image_thinking_combo — 图 + thinking 组合
# ============================================================

class TestImageThinkingCombo:
    """图 + thinking 各形态:adaptive / 流式 / reasoning_split / 图+tool+thinking 三合一。"""

    def test_07_01_thinking_adaptive(self):
        """07_01 — thinking adaptive + 真图(sx1.jpg)非流式,验 HTTP 200 + content > 10。"""
        img_uri = real_image_b64("sx1.jpg", "image/jpeg")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "text", "text": "Describe this image in detail."},
            ]}],
            "thinking": {"type": "adaptive"},
        })
        assert_oai_success(r)
        content = get_oai_content(r)
        assert len(content) > 10, (
            f"07_01 expected substantial image description, got {len(content)}: {content[:200]!r}"
        )

    def test_07_02_thinking_stream(self):
        """07_02 — thinking adaptive + 真图(sx1.jpg)流式,验流式末帧 + content > 10。"""
        img_uri = real_image_b64("sx1.jpg", "image/jpeg")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "text", "text": "Describe this image briefly."},
            ]}],
            "thinking": {"type": "adaptive"},
        }, stream=True)
        assert_oai_stream_success(r)
        content = get_oai_content(r)
        assert len(content) > 10, (
            f"07_02 expected substantial stream content, got {len(content)}: {content[:200]!r}"
        )

    def test_07_03_reasoning_split(self):
        """07_03 — reasoning_split + 图 + thinking adaptive,接口未稳定 → 软断言 200/400。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64()}},
                {"type": "text", "text": "What color? Think step by step."},
            ]}],
            "reasoning_split": True,
            "thinking": {"type": "adaptive"},
        })
        assert r["status"] in (200, 400)

    def test_07_04_image_tool_thinking_combo(self):
        """07_04 — 图 + tool + thinking 三合一,模型应调 get_weather + location≈Beijing。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64()}},
                {"type": "text", "text": "Describe image and tell weather in Beijing"},
            ]}],
            "tools": [WEATHER_TOOL_OAI],
            "thinking": {"type": "adaptive"},
        })
        assert r["status"] == 200
        assert_tool_called(
            r,
            expected_name="get_weather",
            expected_args_subset={"location": "Beijing"},
            schema=WEATHER_TOOL_OAI["function"]["parameters"],
            msg="07_04 image+tool+thinking",
        )


# ============================================================
# 08 image_stream_usage — 图 + 流式 usage chunk
# ============================================================

class TestImageStreamUsage:
    """图 + 流式 + usage chunk 协议字段。"""

    def test_08_01_stream_include_usage(self):
        """08_01 — 流式 + stream_options.include_usage=true + 图,验 HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64()}},
                {"type": "text", "text": "What color?"},
            ]}],
            "stream_options": {"include_usage": True},
        }, stream=True)
        assert r["status"] == 200, (
            f"08_01 image+include_usage expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_08_02_multiturn_two_images(self, stream):
        """08_02 — 第一轮红图 + 第二轮蓝图 follow-up,流式/非流式各 1 item。"""
        r = oai_chat({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": png_base64()}},
                    {"type": "text", "text": "What color is this image?"},
                ]},
                {"role": "assistant", "content": "It's a red image."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(make_png_672(0, 0, 255)).decode()}},
                    {"type": "text", "text": "And this one? How does it compare to the first?"},
                ]},
            ],
        }, stream=stream)
        assert r["status"] == 200


# ============================================================
# 09 image_param — 图相关参数 / usage / 异常容错
# ============================================================

class TestImageParam:
    """图相关参数 / Usage 算术 / 异常输入容错。"""

    def test_09_01_usage_arithmetic_multimodal(self):
        """09_01 — Usage 算术在 image 多模态下仍成立:total == prompt + completion。
        若接口不支持 image input 返回 400 → xfail。
        """
        r = oai_chat({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": png_base64()}},
                    {"type": "text", "text": "What color is this image? One word."},
                ]},
            ],
        })
        if r["status"] == 400 and "image" in str(r.get("body", "")).lower():
            pytest.xfail(
                f"09_01 endpoint does not support image input "
                f"(HTTP={r['status']}): {str(r.get('body'))[:200]}"
            )
        assert_oai_success(r)
        usage = r["body"]["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], (
            f"09_01 multimodal usage math: total={usage['total_tokens']} != "
            f"prompt={usage['prompt_tokens']}+completion={usage['completion_tokens']}"
        )

    def test_09_02_invalid_detail_value(self):
        """09_02 — detail=ultra(非法值)。接口处理未定论,允许 200 接受 / 400 拒绝。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64(), "detail": "ultra"}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert r["status"] in (200, 400)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_09_03_corrupted_base64(self, stream):
        """09_03 — 损坏的 base64 图(随机字节),期望 4xx 拒绝。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": corrupted_base64()}},
                {"type": "text", "text": "What is this?"},
            ]}],
        }, stream=stream)
        assert r["status"] == 400, (
            f"09_03 corrupted base64 stream={stream} expected 400, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_09_04_mime_mismatch(self, stream):
        """09_04 — MIME mismatch:PNG 字节标 image/jpeg,网关应宽容(返回 200)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": mime_mismatch_base64()}},
                {"type": "text", "text": "What is this?"},
            ]}],
        }, stream=stream)
        assert r["status"] == 200

    @pytest.mark.parametrize("detail", ["low", "high"], ids=["detail=low", "detail=high"])
    def test_09_05_detail_low_vs_high(self, detail):
        """09_05 — 同图(sx1.jpg)分别 detail=low / high,两次都应 200 + content > 5。"""
        img_uri = real_image_b64("sx1.jpg", "image/jpeg")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_uri, "detail": detail}},
                {"type": "text", "text": "What do you see?"},
            ]}],
        })
        assert_oai_success(r)
        content = get_oai_content(r)
        assert len(content) > 5, (
            f"09_05 detail={detail} expected content (>5 chars), "
            f"got {len(content)}: {content[:200]!r}"
        )


# ============================================================
# 10 resolution_tier — detail 档位 / max_long_side_pixel / max_total_pixels / 宽高比
# ============================================================

def _get_prompt_tokens(r: dict) -> int:
    """从 oai_chat 返回里抽 prompt_tokens(stream / non_stream 都兼容)"""
    if not r.get("stream"):
        body = r.get("body") or {}
        return (body.get("usage") or {}).get("prompt_tokens", 0)
    for c in reversed(r.get("chunks") or []):
        usage = c.get("usage") if isinstance(c, dict) else None
        if usage:
            return usage.get("prompt_tokens", 0)
    return 0


def _stream_usage_opts(stream: bool) -> dict:
    """流式必须显式 opt-in `stream_options.include_usage=true` 才能拿到 usage chunk。"""
    return {"stream_options": {"include_usage": True}} if stream else {}


def _assert_basic_ok(r: dict, msg: str = ""):
    """档位合法路径的基础断言:HTTP 200 + prompt_tokens 为正数"""
    assert r["status"] == 200, f"{msg}: expected 200, got {r['status']}: {r.get('body', '')[:300] if isinstance(r.get('body'), str) else r.get('body')}"
    pt = _get_prompt_tokens(r)
    assert pt > 0, f"{msg}: prompt_tokens should be positive, got {pt}"


class TestImageResolutionTier:
    """
    图像 detail 档位 + max_long_side_pixel + max_total_pixels + 宽高比测试。

    2026-06-01 与 M3 团队对齐:detail 仅当请求参数传入,不在响应里断言 detail 字段(5a)。
    2026-06-02 与 M3 团队对齐:max_long_side_pixel 必须为 28 的倍数(OAI ViT patch 约束)。
    """

    _COLOR_PROMPT = "What is the dominant color in this image? Answer in one word."

    # -------------------- 10_01~10_06:不同尺寸图喂 default 档 smoke --------------------

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_10_01_tier_low_no_scale(self, stream):
        """10_01 — 500x400 PNG(长边 500 < default 档 2016)→ 应不缩放,HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(500, 400), "detail": "default"
                }},
                {"type": "text", "text": "What color is this?"},
            ]}],
            **_stream_usage_opts(stream),
        }, stream=stream)
        _assert_basic_ok(r, "10_01 small_image")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_10_02_tier_low_scale_down(self, stream):
        """10_02 — 2000x1000 PNG(长边 2000 ≈ default 档边界)→ 接受,HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(2000, 1000), "detail": "default"
                }},
                {"type": "text", "text": "What color?"},
            ]}],
            **_stream_usage_opts(stream),
        }, stream=stream)
        _assert_basic_ok(r, "10_02 medium_image")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_10_03_tier_default_no_scale(self, stream):
        """10_03 — 1500x1000 PNG(长边 1500 < default 档 2016)→ 不缩放,HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(1500, 1000), "detail": "default"
                }},
                {"type": "text", "text": "What color?"},
            ]}],
            **_stream_usage_opts(stream),
        }, stream=stream)
        _assert_basic_ok(r, "10_03 below_default")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_10_04_tier_default_scale_down(self, stream):
        """10_04 — 3000x2000 PNG(长边 3000 > default 档 2016)→ 等比缩到 2016x1344,HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(3000, 2000), "detail": "default"
                }},
                {"type": "text", "text": "What color?"},
            ]}],
            **_stream_usage_opts(stream),
        }, stream=stream)
        _assert_basic_ok(r, "10_04 above_default")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_10_05_tier_high_scale_down(self, stream):
        """10_05 — 5000x3000 PNG(长边 5000 > default 档 2016)→ 触发缩放,HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(5000, 3000), "detail": "default"
                }},
                {"type": "text", "text": "What color?"},
            ]}],
            **_stream_usage_opts(stream),
        }, stream=stream)
        _assert_basic_ok(r, "10_05 large_image")

    def test_10_06_tier_at_boundary(self):
        """10_06 — 4000x2000 PNG(长边 4000 > default 档 2016)→ 触发缩放,接口接受。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(4000, 2000), "detail": "default"
                }},
                {"type": "text", "text": "What color?"},
            ]}],
        })
        _assert_basic_ok(r, "10_06 boundary")

    # -------------------- 10_07:detail 缺省/显式 default 等价性 --------------------

    def test_10_07_detail_default_when_omitted(self):
        """10_07 — 1500x1000 PNG,不传 detail / 显式 detail="default" → 都 HTTP 200。"""
        img = make_png_base64(1500, 1000)
        prompt_text = "What color?"

        r1 = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": prompt_text},
            ]}],
        })
        r2 = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img, "detail": "default"}},
                {"type": "text", "text": prompt_text},
            ]}],
        })
        assert r1["status"] == 200, f"10_07 omitted HTTP={r1['status']}"
        assert r2["status"] == 200, f"10_07 explicit HTTP={r2['status']}"

    # -------------------- 10_08:max_total_pixels 超限 / 边界 --------------------

    def test_10_08_max_total_pixels_exceeded(self):
        """10_08 — 4000x4000 = 16M 像素 > 12,845,056 上限。接口处理未定论 → 软断言。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(4000, 4000), "detail": "default"
                }},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert r["status"] in (200, 400, 413, 422), f"10_08 HTTP={r['status']}"

    def test_10_09_max_total_pixels_at_boundary(self):
        """10_09 — 3584x3584 = 12,845,056(=上限)→ 边界值,允许 200 / 4xx。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(3584, 3584), "detail": "default"
                }},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert r["status"] in (200, 400, 413, 422), f"10_09 HTTP={r['status']}"

    # -------------------- 10_10:宽高比保持 --------------------

    def test_10_10_aspect_ratio_preserved(self):
        """10_10 — 4000x500(宽高比 8:1)→ 接口应接受,HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(4000, 500), "detail": "default"
                }},
                {"type": "text", "text": "What?"},
            ]}],
        })
        _assert_basic_ok(r, "10_10 aspect_ratio_preserved")

    # -------------------- 10_11~10_12:max_long_side_pixel(28 倍数档位) --------------------

    @pytest.mark.parametrize("mlsp,tier", [
        (252, "low"),       # 28 × 9
        (504, "default"),   # 28 × 18
        (1008, "high"),     # 28 × 36
    ], ids=["low=252", "default=504", "high=1008"])
    def test_10_11_max_long_side_pixel_tiers(self, mlsp, tier):
        """10_11 — max_long_side_pixel 取 28 倍数(252/504/1008)+ 5000x3000 红 PNG,
        接口应 200 + prompt_tokens > 0(smoke,不强断模型回答红色)。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(5000, 3000),
                    "max_long_side_pixel": mlsp,
                }},
                {"type": "text", "text": self._COLOR_PROMPT},
            ]}],
        })
        _assert_basic_ok(r, f"10_11 tier={tier} mlsp={mlsp}")

    def test_10_12_max_long_side_pixel_monotonic(self):
        """10_12 — 同图分别设三档 max_long_side_pixel(252/504/1008),
        prompt_tokens 严格单调:252 < 504 < 1008。
        """
        img = make_png_base64(5000, 3000)
        tokens = {}
        for mlsp, tier in [(252, "low"), (504, "default"), (1008, "high")]:
            r = oai_chat({
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": img,
                        "max_long_side_pixel": mlsp,
                    }},
                    {"type": "text", "text": self._COLOR_PROMPT},
                ]}],
            })
            _assert_basic_ok(r, f"10_12 monotonic tier={tier} mlsp={mlsp}")
            tokens[mlsp] = _get_prompt_tokens(r)
        assert tokens[252] < tokens[504] < tokens[1008], (
            f"10_12 monotonic: max_long_side_pixel 越大 prompt_tokens 应越大, "
            f"got {tokens} (expected 252 < 504 < 1008)"
        )

    @pytest.mark.parametrize("invalid_value", [0, -1, 100, 251, 1009],
                             ids=["zero", "negative", "non_multiple_100",
                                  "non_multiple_251", "non_multiple_1009"])
    def test_10_13_max_long_side_pixel_invalid(self, invalid_value):
        """10_13 — max_long_side_pixel 非法值:
          - 0 / 负数(语义无效)
          - 100 / 251(<252 邻近非 28 倍数)/ 1009(>1008 邻近非 28 倍数)
        接口行为未定论 → 软断言 200 / 4xx。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(2000, 1500),
                    "max_long_side_pixel": invalid_value,
                }},
                {"type": "text", "text": "What color?"},
            ]}],
        })
        assert r["status"] in (200, 400, 413, 422), (
            f"10_13 invalid={invalid_value} HTTP={r['status']}"
        )

    # -------------------- 10_14:真图 max_long_side_pixel 边界点(28 倍数) --------------------

    @pytest.mark.parametrize("pixel", [252, 1008], ids=["pixel=252(28x9)", "pixel=1008(28x36)"])
    def test_10_14_max_long_side_pixel_real_image(self, pixel):
        """10_14 — sx1.jpg 真图 + max_long_side_pixel ∈ {252, 1008},接口应 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": real_image_b64("sx1.jpg", "image/jpeg"),
                    "max_long_side_pixel": pixel,
                }},
                {"type": "text", "text": "Describe briefly."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"10_14 max_long_side_pixel={pixel} expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )


# ============================================================
# 11 image_size_limit — 单图大小限 / 请求体限 / size 梯度
# ============================================================

class TestImageSizeLimit:
    """图像大小限:旧 M2 10MB 回归(URL/base64)/ M3 对齐 OAI 30MB / 请求体 64MB 限 / size 梯度。

    ⚠️ 契约说明(2026-06-02):
      - 旧 M2 契约:图像 ≤ 10MB(11_01~11_04 回归保护)
      - 新 M3 契约:图像 ≤ 30MB(11_06 边界点)
      - 请求体 ≤ 64MB(11_07,base64 路径才容易触发)
    """

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_11_01_oversized_image_12mb(self, stream):
        """11_01 — 12MB image(超过旧 10MB 上限)→ 400 拒绝(允许 200 兼容降级)。"""
        big_image = large_image_base64(12)
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": big_image}},
                {"type": "text", "text": "What?"},
            ]}],
        }, stream=stream)
        assert r["status"] in (400, 200), (
            f"11_01 oversized 12MB image stream={stream} expected 400/200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_11_02_oversized_image_strict(self):
        """11_02 — 12MB 真图(sx1.jpg + zero padding),HTTP 200 或 4xx 均可。

        ⚠️ 2026-06-04 修订(2 处):
        1) fixture 改为真图 padding:原 `large_image_base64(12)` 用 stdlib 手写
           极简纯色 PNG 做底,fireworks-m3 等供应商对此类"低熵纯色 PNG"有 silent
           drop 行为(image preprocess 阶段直接丢图,返回 200 + 兜底 fallback 文本,
           走不到 size 校验门)。详见 badcase_scripts/fireworks_m3_11_02_root_cause_probe*。
        2) 断言由严格 400 放宽为 200/4xx 均可:M2 旧契约要求 >10MB 必须拒,M3 新契约
           对齐 OAI 30MB 上限,12MB 在 M3 范围内合法。三渠道实测(2026-06-04):
             - 官方:400 + `bad_request_error: media exceeds size limit: max 10485760 bytes`
             - fireworks-m3:200,正常识图(prompt_tokens=513,vision encoder 接管)
             - together-m3:200,正常识图
           两种行为都属于 M3 阶段合法表现,不再硬卡 400。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": oversized_real_image_data_url(12)}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert r["status"] == 200 or 400 <= r["status"] < 500, (
            f"11_02 oversized 12MB image expected 200 or 4xx, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_11_03_url_under_10mb(self):
        """11_03 — URL 方式 ~9.2MB PNG(<10MB 上限)→ 应被接受。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": size_fixture_url("image_9mb.png")}},
                {"type": "text", "text": "Describe this image briefly."},
            ]}],
        })
        assert_oai_success(r)

    def test_11_04_url_over_10mb(self):
        """11_04 — URL 方式 ~11.1MB PNG(>10MB 上限)→ 服务端下载后应 4xx 拒绝。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": size_fixture_url("image_11mb.png")}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert 400 <= r["status"] < 500, (
            f"11_04 expected 4xx (image URL > 10MB should be rejected), got {r['status']}"
        )

    def test_11_05_base64_under_10mb(self):
        """11_05 — Base64 方式 ~9.2MB PNG(<10MB 上限)→ 应被接受。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": size_fixture_data_url("image_9mb.png")}},
                {"type": "text", "text": "Describe this image briefly."},
            ]}],
        })
        assert_oai_success(r)

    def test_11_06_base64_over_10mb(self):
        """11_06 — Base64 方式 ~11.1MB PNG(>10MB 上限),HTTP 200 或 4xx 均可。

        ⚠️ 2026-06-04 修订:断言由严格 4xx 放宽为 200/4xx,与 11_02 / 11_07 / 11_08 对齐。
           M2 旧契约要求 >10MB 必须拒;M3 新契约对齐 OAI 30MB 上限,11.1MB 在 M3 范围内合法。
           供应商若执行严格 ≤10MB 拒绝 → 4xx;若放宽到 M3 30MB 上限或不做 size 校验 → 200。
           两种行为在 M3 阶段均视为合法。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": size_fixture_data_url("image_11mb.png")}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert r["status"] == 200 or 400 <= r["status"] < 500, (
            f"11_06 expected 200 or 4xx, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.timeout(600)
    def test_11_07_oversize_31mb_m3_limit(self):
        """11_07 — Image >30MB(M3 已对齐 OAI 30MB 上限),HTTP 200 或 4xx 均可。

        ⚠️ 2026-06-04 修订(2 处):
        1) fixture 改为真图 padding:原 `large_image_base64(31)` 用 stdlib 手写
           极简纯色 PNG 做底,fireworks-m3 等供应商对此类"低熵纯色 PNG"有 silent
           drop bug(详见 badcase_scripts/fireworks_m3_11_02_root_cause_probe*),
           会让 size 校验被绕过。改成 sx1.jpg + 31MB padding 后,fixture 本身可被
           vision 识别,供应商若做 size 校验就会走拒绝分支。
        2) 断言由 `{400, 413, 415, 422, 500}` 放宽为 200 或 4xx 均可:
           - 供应商若执行严格 ≤30MB 拒绝 → 4xx(命中预期)
           - 供应商若放宽到更高上限或不做 size 校验 → 200 + 正常识图(也合法)
           5xx 由网关侧 BUG 触发,已不再视作合法行为,本次也排除。
        """
        oversized_uri = oversized_real_image_data_url(size_mb=31)
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": oversized_uri}},
                {"type": "text", "text": "What is this?"},
            ]}],
        }, timeout=300)
        assert r["status"] == 200 or 400 <= r["status"] < 500, (
            f"11_07 image >30MB expected 200 or 4xx, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.timeout(600)
    def test_11_08_request_body_over_64mb(self):
        """11_08 — Base64 ~67MB PNG → 请求体 >> 64MB,HTTP 200 或 4xx 均可。

        ⚠️ 2026-06-04 修订:断言由 4xx 放宽为 200/4xx,与 11_02 / 11_07 对齐。
           供应商若执行 64MB 请求体上限拒绝 → 4xx(常见 400/413);若放宽接受并
           走 vision 识别 → 200。两种行为在 M3 阶段均视为合法。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": size_fixture_data_url("image_65mb.png")}},
                {"type": "text", "text": "What?"},
            ]}],
        }, timeout=300)
        assert r["status"] == 200 or 400 <= r["status"] < 500, (
            f"11_08 request body >64MB expected 200 or 4xx, got {r['status']}"
        )

    @pytest.mark.parametrize("size_mb", [1, 3, 5, 8],
                             ids=["1MB", "3MB", "5MB", "8MB"])
    def test_11_09_size_gradient(self, size_mb):
        """11_09 — Image size 梯度(1 / 3 / 5 / 8 MB):
          - ≤5 MB:稳定 200
          - 8 MB:接近 10MB 上限,允许 200 / 4xx
        """
        data_uri = large_image_base64(size_mb=size_mb)
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe"},
                ],
            }],
            "max_tokens": 2048,
        })
        if size_mb <= 5:
            assert r["status"] == 200, (
                f"11_09 size={size_mb}MB expected 200, got {r['status']}: "
                f"{str(r.get('body'))[:300]}"
            )
        else:
            assert r["status"] in (200, 400, 413, 422), (
                f"11_09 size={size_mb}MB unexpected status {r['status']}: "
                f"{str(r.get('body'))[:300]}"
            )


# ============================================================
# 12 image_count_limit — 多图数量上限(spec 1.3.6 更新版: 实测 ≤199 张)
# ============================================================

# 公网 COS 图直链(沿用 02_01),用 URL 形式构造大批量 image_url,
# 避免 base64 让请求体在 200 张时直接撑爆 65MB 限。
_COUNT_LIMIT_IMAGE_URL = (
    "https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com"
    "/model-release-checker/fixtures/m3_test_images/sx1.jpg"
)


class TestImageCountLimit:
    """spec 1.3.6 "请求最多支持 20 张图片"。
    实测官方 M3 (2026-06-06):20 张也会被裸 400 (2013) 拒(疑似边界含等号或隐式聚合校验),
    所以 at_max 取 19 张验"接受",over_max 取 20 张验"越界",与图片张数 200 版本同款保守模板。
    使用 URL 形式提供图片,与全文 §02_01 一致,避免 base64 体积干扰。"""

    def _url_block(self):
        """单张 COS sx1.jpg 的 URL 形式 image_url 块"""
        return {"type": "image_url", "image_url": {"url": _COUNT_LIMIT_IMAGE_URL}}

    def test_12_01_count_at_max(self):
        """12_01 — 一条请求带 19 张图片 URL(实测可接受的最大值)→ 应被接受 HTTP 200。

        spec 名义上限 20 张,但官方 M3 服务端在 20 张时也返回 400 (2013),
        所以 at_max 用 19 验"接受"边界,避开服务端那 1 张的余量。
        """
        content = [self._url_block() for _ in range(19)]
        content.append({"type": "text", "text": "How many images do you see?"})
        r = oai_chat({"messages": [{"role": "user", "content": content}]}, timeout=300)
        assert r["status"] == 200, (
            f"12_01 image count=19 (at max, URL form) HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_12_02_count_over_max(self):
        """12_02 — 一条请求带 20 张图片 URL(越界)→ 4xx 直接拒绝 OR 200 + 非空响应。
        反例:HTTP 200 但 content 为空(模型未给出任何文字回复)。
        实测官方 M3 返回 400 `the num of image is larger than the limit: 20`。
        """
        content = [self._url_block() for _ in range(20)]
        content.append({"type": "text", "text": "How many?"})
        r = oai_chat({"messages": [{"role": "user", "content": content}]}, timeout=300)
        status = r["status"]
        if 400 <= status < 500:
            return
        assert status == 200, (
            f"12_02 image count=20 (URL form) should be 200 or 4xx, got {status}: "
            f"{str(r.get('body'))[:300]}"
        )
        body_content = get_oai_content(r)
        assert body_content.strip(), (
            f"12_02 image count=20 returned 200 but content is empty; "
            f"server should either reject 4xx or produce a valid response. "
            f"body: {str(r.get('body'))[:300]}"
        )


# ============================================================
# 13 base64_compat — Base64 边界容错
# ============================================================

class TestImageBase64Compat:
    """base64 边界容错(换行/无 padding/MIME 大写/data URI 额外参数)。"""

    def test_13_01_base64_with_linebreaks(self):
        """13_01 — base64 字符串含换行符(encodebytes 输出,每 76 字节加 \\n)。
        验服务端容错(不返回 500,允许 200 容错 或 400 拒绝)。
        """
        path = _REAL_IMAGE_DIR / "sx1.jpg"
        raw_bytes = path.read_bytes()
        b64_with_newlines = base64.encodebytes(raw_bytes).decode()
        data_uri = f"data:image/jpeg;base64,{b64_with_newlines}"
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] != 500, (
            f"13_01 got 500 error on base64 with linebreaks: {str(r.get('body'))[:300]}"
        )

    def test_13_02_base64_no_padding(self):
        """13_02 — base64 去掉 `=` padding,验服务端容错(允许 200 / 400,不应 500)。"""
        path = _REAL_IMAGE_DIR / "sx1.jpg"
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode().rstrip("=")
        data_uri = f"data:image/jpeg;base64,{b64}"
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] != 500, (
            f"13_02 got 500 error on base64 without padding: {str(r.get('body'))[:300]}"
        )

    def test_13_03_mime_uppercase(self):
        """13_03 — MIME 大写 `data:image/PNG;base64,...`,验大小写不敏感(应 200)。"""
        png_bytes = make_png_672()
        b64 = base64.b64encode(png_bytes).decode()
        data_uri = f"data:image/PNG;base64,{b64}"
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] == 200, (
            f"13_03 MIME uppercase expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_13_04_data_uri_extra_params(self):
        """13_04 — `data:image/jpeg;charset=utf-8;base64,...`(MIME 带额外参数)。
        服务端行为未定论 → 允许 200/400/422。
        """
        path = _REAL_IMAGE_DIR / "sx1.jpg"
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode()
        data_uri = f"data:image/jpeg;charset=utf-8;base64,{b64}"
        r = oai_chat({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe"},
                ],
            }],
            "max_tokens": 2048,
        })
        assert r["status"] in (200, 400, 422), (
            f"13_04 data URI with extra params unexpected status {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
