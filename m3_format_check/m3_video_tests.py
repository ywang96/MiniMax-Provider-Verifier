"""
M3 API Test — 视频模态 case 集合

按"校验内容"分模块组织,case 命名规范:
    test_<模块编号>_<模块内顺序编号>_<场景说明>

模块编号 / 主题:
    01  base64_video          base64 视频基础接受性
    02  url_video             URL 视频接受性
    03  video_format          视频容器/MIME 格式(MOV/MKV/AVI 等)
    04  multi_video           多段视频叠加 / 数量上限
    05  image_video_mixed     图 + 视频混合 message
    06  fps_param             fps 合法档位 与 越界
    07  detail_param          detail / fps 字段缺省与组合
    08  resolution_tier       分辨率档位 / 边界 / 像素上限
    09  max_long_side_pixel   max_long_side_pixel(28 倍数)契约
    10  video_size_limit      视频大小限(≤50MB)
    11  long_video            长视频(5/10/20/30 min)
    12  media_gradient        分辨率梯度 / 多视频梯度
    13  video_extension       reasoning_split 等扩展字段
    14  error_codes           视频相关错误码

模态优先级 video > image > text,本文件含视频(含图片+视频的混合归本文件)。

所有 case 透过 helpers.oai_chat() 走 /v1/chat/completions,jsonl 落
RUN_LOG_PATH(由 conftest 注入)。
"""
import base64
import os
from pathlib import Path

import pytest

from helpers import *
from image_tools import (
    fixture_video_base64,
    fixture_video_url,
    VIDEO_FIXTURES,
)


# ============================================================
# 本文件局部 helper:真实视频 fixture 读取
# (避免污染 helpers.py / 与 M3 团队参考实现的 real_video_b64 同名同义)
# ============================================================

_REAL_VIDEO_DIR = Path(__file__).parent / "fixtures" / "m3_test_videos"
_REAL_IMAGE_DIR = Path(__file__).parent / "fixtures" / "m3_test_images" / "real"


def real_video_b64(name: str = "real_2s.mp4", mime: str = "video/mp4") -> str:
    """读取 fixtures/m3_test_videos/<name> → base64 data URL。

    与 m3_image_tests.py:real_image_b64 同套语义。常用 fixture:
      - real_2s.mp4     (~104KB,P09/P14 内容理解 case)
      - 12s_real.mp4    (~628KB,06 fps 合法档位 12 秒视频)
      - flower-video.mp4 (~1MB,图+视频混合 case 第二段)
      - test_video.mov / .mkv / .avi (~160KB,03 模块真实非 MP4 格式 smoke)
    """
    path = _REAL_VIDEO_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"real video fixture missing: {path}\n"
            f"参考实现期望 fixtures/m3_test_videos/ 下存在 real_2s.mp4 / 12s_real.mp4 等"
        )
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def real_image_b64_for_video(name: str = "sx1.jpg", mime: str = "image/jpeg") -> str:
    """读 fixtures/m3_test_images/real/<name> 的本地副本,供视频文件里的图+视频混合 case 用。

    与 m3_image_tests.py:real_image_b64 隔离,避免跨文件 import。
    """
    path = _REAL_IMAGE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"real image fixture missing: {path}\n"
            f"参考实现期望 fixtures/m3_test_images/real/ 下存在 sx1.jpg"
        )
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


# 长视频(>10MB)优先从 M3_LONG_VIDEO_DIR 环境变量指定的目录读取;
# 未设置时 fallback 到本地 fixtures/m3_test_videos/(D_300s/D_600s/D_1200s/D_1800s.mp4 已入仓)。
# 例如:export M3_LONG_VIDEO_DIR=/Users/minimax/Downloads/M3/assets/videos/duration
_LONG_VIDEO_DIR_ENV = "M3_LONG_VIDEO_DIR"
_LOCAL_LONG_DIR = Path(__file__).parent / "fixtures" / "m3_test_videos"


def long_video_path(name: str) -> Path:
    base = os.environ.get(_LONG_VIDEO_DIR_ENV)
    if base:
        return Path(base) / name
    return _LOCAL_LONG_DIR / name


def long_video_b64(name: str, mime: str = "video/mp4") -> str:
    p = long_video_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"long video missing: {p}\n"
            f"set {_LONG_VIDEO_DIR_ENV}=<dir-containing-{name}>"
        )
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


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


def _assert_basic_ok(r: dict, msg: str = ""):
    """档位合法路径的基础断言:HTTP 200 + prompt_tokens 为正数"""
    assert r["status"] == 200, (
        f"{msg}: expected 200, got {r['status']}: "
        f"{r.get('body', '')[:300] if isinstance(r.get('body'), str) else r.get('body')}"
    )
    pt = _get_prompt_tokens(r)
    assert pt > 0, f"{msg}: prompt_tokens should be positive, got {pt}"


def _video_payload(filename: str, detail: str = None, fps: float = None,
                   text: str = "What is this?") -> dict:
    """构造视频请求的 video_url 块"""
    video_url_obj = {"url": fixture_video_base64(filename)}
    if detail is not None:
        video_url_obj["detail"] = detail
    if fps is not None:
        video_url_obj["fps"] = fps
    return {
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": video_url_obj},
            {"type": "text", "text": text},
        ]}],
    }


# 多个模块共用的真实素材标记(春节卡通小马,5MB)
PONY_VIDEO_FIXTURE = "476117246902419462.mp4"
PONY_VIDEO_URL = (
    "https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/"
    "476117246902419462.mp4"
)


# ============================================================
# 01 base64_video — base64 视频基础接受性
# ============================================================

class TestBase64Video:
    """base64 dataURL 形式喂视频:验最小可用路径 + 真实素材内容理解。"""

    def test_01_01_base64_video(self):
        """base64 最小 MP4 + What do you see → HTTP 200 + 非空回答。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": mp4_base64()}},
                {"type": "text", "text": "What do you see?"},
            ]}],
        })
        assert r["status"] == 200
        content = get_oai_content(r)
        assert len(content.strip()) > 20, (
            f"01_01 video understanding should yield non-trivial response, "
            f"got len={len(content)}: {content!r}"
        )

    def test_01_02_base64_real_pony_cartoon(self):
        """真实素材(春节卡通小马 5MB)base64 输入,验内容理解 + content > 50。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": fixture_video_base64(PONY_VIDEO_FIXTURE),
                }},
                {"type": "text", "text": "这个视频里有什么?请详细描述。"},
            ]}],
            "max_tokens": 600,
        })
        assert r["status"] == 200, (
            f"01_02 video base64 should return 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert len(content.strip()) > 50, (
            f"01_02 expected non-trivial response, got len={len(content)}: {content!r}"
        )


# ============================================================
# 02 url_video — URL 视频接受性
# ============================================================

class TestURLVideo:
    """video_url.url 走公网/COS URL 形式。"""

    def test_02_01_url_video(self):
        """公网示例 mp4 URL → HTTP 200 + 非空回答(基础 URL 接受性)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": SAMPLE_VIDEO_URL}},
                {"type": "text", "text": "What is in this video?"},
            ]}],
        })
        assert r["status"] == 200
        content = get_oai_content(r)
        assert len(content.strip()) > 20, (
            f"02_01 video understanding should yield non-trivial response, "
            f"got len={len(content)}: {content!r}"
        )

    def test_02_02_url_real_pony_cartoon(self):
        """真实素材(春节卡通小马)经 OSS URL 输入,验内容理解 + content > 50。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": PONY_VIDEO_URL}},
                {"type": "text", "text": "这个视频里有什么?请详细描述。"},
            ]}],
            "max_tokens": 600,
        })
        assert r["status"] == 200, (
            f"02_02 video_url should return 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
        content = get_oai_content(r)
        assert len(content.strip()) > 50, (
            f"02_02 expected non-trivial response, got len={len(content)}: {content!r}"
        )


# ============================================================
# 03 video_format — 视频容器/MIME 格式(MOV/MKV/AVI 等)
# ============================================================

class TestVideoFormat:
    """视频格式兼容性 — 含伪装 MIME(MOV/MKV/AVI 等)与真实非 MP4 文件 smoke。"""

    def test_03_01_mkv_format_legacy(self):
        """MKV(video/x-matroska)用最小 mp4 伪装 → 已知不支持时 xfail。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": "data:video/x-matroska;base64,"
                           + base64.b64encode(make_minimal_mp4_504()).decode(),
                }},
                {"type": "text", "text": "What is this?"},
            ]}],
        })
        if r["status"] != 200:
            pytest.xfail("Known BUG: MKV format not supported")

    def test_03_02_mov_format_video_quicktime(self):
        """MOV(video/quicktime base64)→ 已知 BUG-7 时 xfail。"""
        mov_b64 = ("data:video/quicktime;base64,"
                   + base64.b64encode(make_minimal_mp4_504()).decode())
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": mov_b64}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        if r["status"] != 200:
            pytest.xfail("Known BUG-7: MOV format not supported")

    def test_03_03_mov_format_video_mov(self):
        """MOV(spec 1.3.5 要求 data:video/mov;base64,...)→ 已知 BUG-7 时 xfail。"""
        mov_b64 = ("data:video/mov;base64,"
                   + base64.b64encode(make_minimal_mp4_504()).decode())
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": mov_b64}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        if r["status"] != 200:
            pytest.xfail("Known BUG-7: MOV format not supported")

    def test_03_04_mkv_format_video_matroska(self):
        """MKV(video/x-matroska)用最小 mp4 伪装 → 已知 BUG-7 时 xfail。"""
        mkv_b64 = ("data:video/x-matroska;base64,"
                   + base64.b64encode(make_minimal_mp4_504()).decode())
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": mkv_b64}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        if r["status"] != 200:
            pytest.xfail("Known BUG-7: MKV format not supported")

    @pytest.mark.parametrize("mime", ["video/avi", "video/x-msvideo"],
                             ids=["video_avi", "video_x_msvideo"])
    def test_03_05_avi_format(self, mime):
        """AVI 两个合法 MIME(video/avi / video/x-msvideo)用最小 mp4 伪装 → 不支持时 xfail。"""
        avi_b64 = f"data:{mime};base64," + base64.b64encode(make_minimal_mp4_504()).decode()
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": avi_b64}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        if r["status"] != 200:
            pytest.xfail(f"AVI format with MIME={mime} not supported / fixture is mp4 data")

    def test_03_06_real_mov_smoke(self):
        """真实 MOV 文件(fixtures/test_video.mov)默认 detail/fps → 接受或合法拒绝。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("test_video.mov", "video/mov"),
                }},
                {"type": "text", "text": "One word."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] in (200, 400, 415, 422, 500), (
            f"03_06 real MOV unexpected {r['status']}: {str(r.get('body'))[:300]}"
        )

    def test_03_07_real_avi_smoke(self):
        """真实 AVI 文件(fixtures/test_video.avi)默认 detail/fps → 接受或合法拒绝。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("test_video.avi", "video/avi"),
                }},
                {"type": "text", "text": "One word."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] in (200, 400, 415, 422, 500), (
            f"03_07 real AVI unexpected {r['status']}: {str(r.get('body'))[:300]}"
        )

    def test_03_08_real_mkv_smoke(self):
        """真实 MKV 文件(fixtures/test_video.mkv)默认 detail/fps → 接受或合法拒绝。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("test_video.mkv", "video/x-matroska"),
                }},
                {"type": "text", "text": "One word."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] in (200, 400, 415, 422, 500), (
            f"03_08 real MKV unexpected {r['status']}: {str(r.get('body'))[:300]}"
        )


# ============================================================
# 04 multi_video — 多段视频叠加 / 数量上限
# ============================================================

class TestMultiVideo:
    """spec 1.3.6 (更新版) 视频数量上限 20 段 + 多段叠加边界覆盖。
    20 段 base64 会撑爆请求体,边界 case 改用 URL 形式 (fixture_video_url)。"""

    def _mp4_block(self):
        return {"type": "video_url", "video_url": {
            "url": "data:video/mp4;base64,"
                   + base64.b64encode(make_minimal_mp4_504()).decode(),
        }}

    def _mp4_url_block(self):
        """单段 video_400x300.mp4 (COS 直链) 的 URL 形式 video_url 块。
        用最小分辨率减小网关下载压力。
        """
        return {"type": "video_url", "video_url": {
            "url": fixture_video_url("video_400x300.mp4"),
        }}

    def test_04_01_count_at_max(self):
        """20 段视频 URL(spec 更新版上限)→ 接口应接受(HTTP 200)。"""
        content = [self._mp4_url_block() for _ in range(20)]
        content.append({"type": "text", "text": "How many videos do you see?"})
        r = oai_chat({"messages": [{"role": "user", "content": content}]}, timeout=600)
        assert r["status"] == 200, (
            f"04_01 video count=20 (at max, URL form) HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_04_02_count_over_max(self):
        """21 段视频 URL(> spec 更新版上限 20)→ 4xx 直接拒绝 OR 200 + 非空响应。
        反例:HTTP 200 但 content 为空(模型未给出任何文字回复)。
        实测官方 M3 返回 400 `the num of video is larger than the limit: 20`(强制拦截);
        部分供应商可能放过,允许 200 + 有内容亦视为合规(贴齐 image 12_02 的写法)。
        """
        content = [self._mp4_url_block() for _ in range(21)]
        content.append({"type": "text", "text": "How many?"})
        r = oai_chat({"messages": [{"role": "user", "content": content}]}, timeout=600)
        status = r["status"]
        if 400 <= status < 500:
            return
        assert status == 200, (
            f"04_02 video count=21 (URL form) should be 200 or 4xx, got {status}: "
            f"{str(r.get('body'))[:300]}"
        )
        resp_content = get_oai_content(r)
        assert resp_content and resp_content.strip(), (
            f"04_02 video count=21 returned 200 but content is empty; "
            f"server should either reject 4xx or produce a valid response. "
            f"body: {str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    def test_04_03_multi_video_3(self):
        """3 段 real_2s.mp4 真实视频叠加,允许 200/4xx。"""
        vid = real_video_b64("real_2s.mp4")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": vid}},
                {"type": "video_url", "video_url": {"url": vid}},
                {"type": "video_url", "video_url": {"url": vid}},
                {"type": "text", "text": "Describe each video briefly."},
            ]}],
            "max_tokens": 2048,
        }, timeout=300)
        assert r["status"] in (200, 400, 413, 422), (
            f"04_03 3 videos unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    def test_04_04_multi_video_5(self):
        """5 段 real_2s.mp4 真实视频叠加(spec 上限),允许 200/4xx。"""
        vid = real_video_b64("real_2s.mp4")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                *[{"type": "video_url", "video_url": {"url": vid}} for _ in range(5)],
                {"type": "text", "text": "Describe each video briefly."},
            ]}],
            "max_tokens": 2048,
        }, timeout=300)
        assert r["status"] in (200, 400, 413, 422), (
            f"04_04 5 videos unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_04_05_multi_video_b64_url_mix(self):
        """混合形态:base64 + URL 两段不同载入形态的多视频,允许 200/4xx。

        与 04_01..04_04 的"全 base64 同段"覆盖不同的输入形态。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "video_url", "video_url": {"url": fixture_video_url("video_400x300.mp4")}},
                {"type": "text", "text": "How many videos do you see?"},
            ]}],
            "max_tokens": 1024,
        }, timeout=300)
        assert r["status"] in (200, 400, 413, 422), (
            f"04_05 multi-video b64+url HTTP={r['status']}: {str(r.get('body'))[:300]}"
        )


# ============================================================
# 05 image_video_mixed — 图 + 视频混合 message
# ============================================================

class TestImageVideoMixed:
    """同 message 内含图 + 视频的混合 multimodal 输入。"""

    def test_05_01_one_image_one_video(self):
        """1 图(PNG)+ 1 视频(real_2s) → HTTP 200(多模态混合)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64()}},
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "text", "text": "Describe both."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"05_01 image+video mixed expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_05_02_image_video_stream_variant(self, stream):
        """图+视频混合 → 非流式/流式两路均应 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_base64()}},
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "text", "text": "Describe both."},
            ]}],
            "max_tokens": 1024,
        }, stream=stream)
        assert r["status"] == 200, (
            f"05_02 stream={stream} mixed image+video expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    def test_05_03_mixed_3img_1vid(self):
        """3 真图(sx1×3) + 1 真实视频(real_2s) 混合 → 允许 200/4xx。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": real_image_b64_for_video("sx1.jpg", "image/jpeg")}},
                {"type": "image_url", "image_url": {"url": real_image_b64_for_video("sx1.jpg", "image/jpeg")}},
                {"type": "image_url", "image_url": {"url": real_image_b64_for_video("sx1.jpg", "image/jpeg")}},
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "text", "text": "Describe all the images and the video."},
            ]}],
            "max_tokens": 2048,
        }, timeout=300)
        assert r["status"] in (200, 400, 413, 422), (
            f"05_03 3img+1vid unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    def test_05_04_mixed_1img_2vid(self):
        """1 真图(sx1) + 2 真实视频(real_2s + flower-video) 混合 → 允许 200/4xx。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": real_image_b64_for_video("sx1.jpg", "image/jpeg")}},
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "video_url", "video_url": {"url": real_video_b64("flower-video.mp4")}},
                {"type": "text", "text": "Describe the image and both videos."},
            ]}],
            "max_tokens": 2048,
        }, timeout=300)
        assert r["status"] in (200, 400, 413, 422), (
            f"05_04 1img+2vid unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )


# ============================================================
# 06 fps_param — fps 合法档位 与 越界
# ============================================================

class TestFpsParam:
    """fps ∈ [0.2, 5] 合法档位 与 越界拒绝行为。"""

    @pytest.mark.parametrize("fps", [0.5, 1, 2], ids=["fps=0.5", "fps=1", "fps=2"])
    def test_06_01_fps_real_video(self, fps):
        """fps ∈ {0.5, 1, 2} 用 12s_real.mp4 真实视频 → HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("12s_real.mp4"),
                    "fps": fps,
                }},
                {"type": "text", "text": "Describe briefly."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"06_01 fps={fps} expected 200, got {r['status']}: {str(r.get('body'))[:300]}"
        )

    @pytest.mark.parametrize("fps", [0.5, 1.0, 2.0, 5.0],
                             ids=["fps=0.5", "fps=1.0", "fps=2.0", "fps=5.0"])
    def test_06_02_valid_fps(self, fps):
        """合法 fps ∈ [0.2, 5] 各档,用 12s_real.mp4 → HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("12s_real.mp4"),
                    "fps": fps,
                }},
                {"type": "text", "text": "Describe."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"06_02 valid fps={fps} expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.parametrize("fps", [0.1, 0.0, -1, 10.0, 100],
                             ids=["fps=0.1", "fps=0", "fps=-1", "fps=10", "fps=100"])
    def test_06_03_invalid_fps(self, fps):
        """越界值(< 0.2 或 > 5),服务端可能拒绝或宽松接受,软断言 200/400/422/500。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("12s_real.mp4"),
                    "fps": fps,
                }},
                {"type": "text", "text": "Describe."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] in (200, 400, 422, 500), (
            f"06_03 invalid fps={fps} unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_06_04_fps_lower_boundary(self):
        """fps=0.2(下界)→ HTTP 200 + prompt_tokens > 0。"""
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=0.2))
        _assert_basic_ok(r, "06_04 fps=0.2 lower boundary")

    def test_06_05_fps_upper_boundary(self):
        """fps=5.0(上界)→ HTTP 200 + prompt_tokens > 0。"""
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=5.0))
        _assert_basic_ok(r, "06_05 fps=5.0 upper boundary")

    @pytest.mark.parametrize("fps", [0.1, 0.0, -1], ids=["0.1", "0.0", "-1"])
    def test_06_06_fps_below_min_short_fixture(self, fps):
        """fps<0.2 用 video_640x480.mp4(短 fixture)→ 软断言 200/400/422,记录实际行为。

        与 06_03 互补:06_03 用 12s_real.mp4(长视频)覆盖完整越界集合,
        06_06 用短 fixture 单独覆盖下界附近的 3 个常见越界值,便于快速诊断。
        """
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=fps))
        assert r["status"] in (200, 400, 422), f"06_06 fps={fps} HTTP={r['status']}"

    @pytest.mark.parametrize("fps", [5.1, 10, 1000], ids=["5.1", "10", "1000"])
    def test_06_07_fps_above_max_short_fixture(self, fps):
        """fps>5 用 video_640x480.mp4(短 fixture)→ 软断言 200/400/422,记录实际行为。

        与 06_03 互补:06_07 覆盖上界附近的 3 个常见越界值,短 fixture 便于快速诊断。
        """
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=fps))
        assert r["status"] in (200, 400, 422), f"06_07 fps={fps} HTTP={r['status']}"


# ============================================================
# 07 detail_param — detail / fps 字段缺省与组合
# ============================================================

class TestDetailParam:
    """detail = low/default/high 三档接受性 + detail / fps 字段缺省组合。"""

    @pytest.mark.parametrize("detail", ["low", "default", "high"],
                             ids=["detail=low", "detail=default", "detail=high"])
    def test_07_01_detail_value(self, detail):
        """video_url.detail ∈ {low, default, high} → HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("real_2s.mp4"),
                    "detail": detail,
                }},
                {"type": "text", "text": "What is in the video?"},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"07_01 detail={detail} expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_07_02_no_detail_no_fps(self):
        """不传 detail / 不传 fps,服务端走默认(detail=default, fps=1)→ HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("real_2s.mp4"),
                }},
                {"type": "text", "text": "What is in the video? One word."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"07_02 默认参数 expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_07_03_no_detail_explicit_fps(self):
        """只传 fps,不传 detail,服务端走默认 detail=default → HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("real_2s.mp4"),
                    "fps": 1,
                }},
                {"type": "text", "text": "One word."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"07_03 仅 fps expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_07_04_no_fps_explicit_detail(self):
        """只传 detail,不传 fps,服务端走默认 fps=1 → HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("real_2s.mp4"),
                    "detail": "default",
                }},
                {"type": "text", "text": "One word."},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"07_04 仅 detail expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_07_05_max_long_side_pixel_baseline(self):
        """max_long_side_pixel=504(28×18 最小档)+ real_2s.mp4 → HTTP 200(契约最小档冒烟)。

        # OAI 契约:max_long_side_pixel 必须为 28 的倍数(2026-06-02 由 M3 团队对齐)。
        # 此 case 仅做最小档冒烟,系列契约/单调/越界完整覆盖在 §09 模块。
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": real_video_b64("real_2s.mp4"),
                    "max_long_side_pixel": 504,
                }},
                {"type": "text", "text": "What is in the video?"},
            ]}],
            "max_tokens": 1024,
        })
        assert r["status"] == 200, (
            f"07_05 max_long_side_pixel=504 expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )


# ============================================================
# 08 resolution_tier — 分辨率档位 / 边界 / 像素上限
# ============================================================

class TestResolutionTier:
    """不同分辨率视频走 detail=default 缩放 + max_total_pixels 软上限。"""

    def test_08_01_tier_low_no_scale(self):
        """400x300 视频(长边 400 < default 档 672)→ HTTP 200(不缩)。"""
        r = oai_chat(_video_payload("video_400x300.mp4", detail="default"))
        _assert_basic_ok(r, "08_01 small_video")

    def test_08_02_tier_low_scale_down(self):
        """1280x720 视频(长边 1280 > default 档 672)→ 应每帧缩,HTTP 200。"""
        r = oai_chat(_video_payload("video_1280x720.mp4", detail="default"))
        _assert_basic_ok(r, "08_02 medium_video")

    def test_08_03_tier_default_no_scale(self):
        """640x480 视频(长边 640 < default 档 672)→ HTTP 200(不缩)。"""
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default"))
        _assert_basic_ok(r, "08_03 below_default")

    def test_08_04_tier_default_scale_down(self):
        """1920x1080 视频(长边 1920 > default 档 672)→ 应每帧缩,HTTP 200。"""
        r = oai_chat(_video_payload("video_1920x1080.mp4", detail="default"))
        _assert_basic_ok(r, "08_04 above_default")

    def test_08_05_tier_high_no_scale(self):
        """1280x720 视频(长边 1280 > default 档 672)→ HTTP 200。"""
        r = oai_chat(_video_payload("video_1280x720.mp4", detail="default"))
        _assert_basic_ok(r, "08_05 medium_large_video")

    def test_08_06_tier_high_scale_down(self):
        """3840x2160 视频(长边 3840 > default 档 672)→ 应每帧缩,HTTP 200。"""
        r = oai_chat(_video_payload("video_3840x2160.mp4", detail="default"))
        _assert_basic_ok(r, "08_06 large_video")

    def test_08_07_tier_at_boundary(self):
        """1280x720 视频(长边 = default 档两倍)→ 边界软断言 HTTP 200。"""
        r = oai_chat(_video_payload("video_1280x720.mp4", detail="default"))
        _assert_basic_ok(r, "08_07 boundary")

    def test_08_08_tier_consistency(self):
        """1920x1080 大视频 + default 档 → HTTP 200 + prompt_tokens > 0(基础冒烟)。"""
        r = oai_chat(_video_payload("video_1920x1080.mp4", detail="default"))
        assert r["status"] == 200, f"08_08 HTTP={r['status']}"
        pt = _get_prompt_tokens(r)
        assert pt > 0, f"08_08 prompt_tokens={pt}"

    def test_08_09_max_total_pixels_exceeded(self):
        """1 秒 3840x2160 + fps=5,逼近 max_total_pixels=301,056,000 软上限 → 200/4xx。

        # 当前 fixtures 都是 1 秒视频不足以真正触发上限,仅做"高 fps 大分辨率"接受性
        # 断言;真正超限测试需更长时长 fixture。
        """
        r = oai_chat(_video_payload("video_3840x2160.mp4", detail="default", fps=5))
        assert r["status"] in (200, 400, 413, 422), (
            f"08_09 HTTP={r['status']}"
        )


# ============================================================
# 09 max_long_side_pixel — max_long_side_pixel(28 倍数)契约
# ============================================================
# OAI 契约:max_long_side_pixel 必须为 28 的倍数(2026-06-02 由 M3 团队对齐)。
# 视频三档采用 504(28×18) / 1008(28×36) / 2016(28×72),
# 旧的 (504, 672, 1280) 组合里 672/1280 不是 28 的倍数,已废弃。

class TestMaxLongSidePixel:
    """max_long_side_pixel 三档(504/1008/2016)契约 + 单调 + 越界软断言。"""

    @pytest.mark.parametrize("mlsp,tier", [
        (504, "low"),
        (1008, "default"),
        (2016, "high"),
    ], ids=["low=504", "default=1008", "high=2016"])
    def test_09_01_tier_value(self, mlsp, tier):
        """三档(504/1008/2016)各档喂 3840x2160 真实视频(长边 > 2016,均触发缩放)→ HTTP 200 + prompt_tokens > 0。"""
        fixture_name = "video_3840x2160.mp4"
        video_url_obj = {
            "url": fixture_video_base64(fixture_name),
            "max_long_side_pixel": mlsp,
        }
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": video_url_obj},
                {"type": "text", "text": "What is this?"},
            ]}],
        })
        _assert_basic_ok(r, f"09_01 tier={tier} mlsp={mlsp} [{fixture_name}]")

    def test_09_02_monotonic(self):
        """同视频分别设三档,prompt_tokens 严格单调 504 < 1008 < 2016。"""
        fixture_name = "video_3840x2160.mp4"
        fixture_url = fixture_video_base64(fixture_name)
        tokens = {}
        for mlsp, tier in [(504, "low"), (1008, "default"), (2016, "high")]:
            r = oai_chat({
                "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {
                        "url": fixture_url,
                        "max_long_side_pixel": mlsp,
                    }},
                    {"type": "text", "text": "What is this?"},
                ]}],
            })
            _assert_basic_ok(r, f"09_02 monotonic tier={tier} mlsp={mlsp}")
            tokens[mlsp] = _get_prompt_tokens(r)
        assert tokens[504] < tokens[1008] < tokens[2016], (
            f"09_02 monotonic: max_long_side_pixel 越大 prompt_tokens 应越大, "
            f"got {tokens} (expected 504 < 1008 < 2016)"
        )

    @pytest.mark.parametrize("invalid_value", [140, 3612, 0, -28],
                             ids=["below_min_28x5", "above_max_28x129", "zero", "negative_28"])
    def test_09_03_out_of_range(self, invalid_value):
        """[150, 3584] spec 范围外(<150 / >3584 / 0 / 负数,均为 28 倍数)→ 软断言 200/400/413/422。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": fixture_video_base64("video_640x480.mp4"),
                    "max_long_side_pixel": invalid_value,
                }},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert r["status"] in (200, 400, 413, 422), (
            f"09_03 invalid={invalid_value} HTTP={r['status']}"
        )


# ============================================================
# 10 video_size_limit — 视频大小限(≤50MB)
# ============================================================

class TestVideoSizeLimit:
    """视频 ≤ 50 MB,URL/Base64 两种方式各自覆盖通过/拒绝两侧,以及 padded 等价物。"""

    def test_10_01_url_under_50mb(self):
        """URL 方式: ~47.4 MB MP4 在 50 MB 上限内,应被接受。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": size_fixture_url("video_49mb.mp4")}},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
        })
        assert_oai_success(r)

    def test_10_02_url_over_50mb(self):
        """URL 方式: ~52 MB MP4 超过 50 MB 上限,服务端下载后应拒绝(4xx)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": size_fixture_url("video_51mb.mp4")}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert 400 <= r["status"] < 500, (
            f"10_02 expected 4xx (video URL > 50MB should be rejected), got {r['status']}"
        )

    @pytest.mark.timeout(600)
    def test_10_03_base64_under_50mb(self):
        """Base64 方式: ~47.4 MB MP4 在 50 MB 上限内,应被接受。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": size_fixture_data_url("video_49mb.mp4")}},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
        }, timeout=300)
        assert_oai_success(r)

    @pytest.mark.timeout(600)
    def test_10_04_base64_over_50mb(self):
        """Base64 方式: ~52 MB MP4 超过 50 MB 上限,应被拒绝(4xx)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": size_fixture_data_url("video_51mb.mp4")}},
                {"type": "text", "text": "What?"},
            ]}],
        }, timeout=300)
        assert 400 <= r["status"] < 500, (
            f"10_04 expected 4xx (video base64 > 50MB should be rejected), got {r['status']}"
        )

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_10_05_padded_over_50mb_rejected(self):
        """real_2s.mp4 + null padding 到 51MB(真实开头 + null 填充)→ 应被拒绝(4xx 或 500)。

        与 10_04 干净 52MB fixture 互补,覆盖不同形态的"超大视频"输入。
        """
        raw = (Path(__file__).parent / "fixtures" / "m3_test_videos" / "real_2s.mp4").read_bytes()
        target_size = 51 * 1024 * 1024
        padded = raw + b"\x00" * (target_size - len(raw))
        data_uri = "data:video/mp4;base64," + base64.b64encode(padded).decode()

        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": data_uri}},
                {"type": "text", "text": "What is this video?"},
            ]}],
            "max_tokens": 1024,
        }, timeout=300)
        assert r["status"] in (400, 413, 415, 422, 500), (
            f"10_05 expected reject for >50MB padded video, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )


# ============================================================
# 11 long_video — 长视频(5/10/20/30 min)
# ============================================================
# fixtures 不入 git(单个文件 > 30MB),通过环境变量
# `M3_LONG_VIDEO_DIR` 指定目录(默认参考实现位于
# /Users/minimax/Downloads/M3/assets/videos/duration/)。

def _long_video_present(name: str) -> bool:
    return long_video_path(name).exists()


class TestLongVideo:
    """5/10/20/30 分钟视频耗时与上限测试 (skipif env)。"""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _long_video_present("D_300s.mp4"),
        reason="long video fixture D_300s.mp4 not present; "
               "set M3_LONG_VIDEO_DIR=<dir-with-D_300s.mp4>",
    )
    def test_11_01_long_video_5min(self):
        """5 分钟视频 (~5MB,默认 fps=1)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": long_video_b64("D_300s.mp4")}},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
            "max_tokens": 2048,
        }, timeout=600)
        assert r["status"] in (200, 400, 413, 422, 500, 502, 503, 504), (
            f"11_01 5min unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _long_video_present("D_600s.mp4"),
        reason="long video fixture D_600s.mp4 not present; "
               "set M3_LONG_VIDEO_DIR=<dir-with-D_600s.mp4>",
    )
    def test_11_02_long_video_10min(self):
        """10 分钟视频 (默认 fps=1)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": long_video_b64("D_600s.mp4")}},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
            "max_tokens": 2048,
        }, timeout=600)
        assert r["status"] in (200, 400, 413, 422, 500, 502, 503, 504), (
            f"11_02 10min unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _long_video_present("D_1200s.mp4"),
        reason="long video fixture D_1200s.mp4 not present; "
               "set M3_LONG_VIDEO_DIR=<dir-with-D_1200s.mp4>",
    )
    def test_11_03_long_video_20min(self):
        """20 分钟视频 (fps=0.5 控总像素)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": long_video_b64("D_1200s.mp4"),
                    "fps": 0.5,
                }},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
            "max_tokens": 2048,
        }, timeout=600)
        assert r["status"] in (200, 400, 413, 422, 500, 502, 503, 504), (
            f"11_03 20min unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _long_video_present("D_1800s.mp4"),
        reason="long video fixture D_1800s.mp4 not present; "
               "set M3_LONG_VIDEO_DIR=<dir-with-D_1800s.mp4>",
    )
    def test_11_04_long_video_30min(self):
        """30 分钟视频 (fps=0.5 控总像素)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": long_video_b64("D_1800s.mp4"),
                    "fps": 0.5,
                }},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
            "max_tokens": 2048,
        }, timeout=600)
        assert r["status"] in (200, 400, 413, 422, 500, 502, 503, 504), (
            f"11_04 30min unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )


# ============================================================
# 12 media_gradient — 分辨率梯度 / 多视频梯度
# ============================================================

class TestMediaGradient:
    """媒体梯度:视频分辨率梯度(1080P/2K)与多视频数量梯度(3/5 段)。"""

    @pytest.mark.parametrize("filename,label", [
        ("video_1920x1080.mp4", "1080P"),
        ("video_3840x2160.mp4", "2K"),
    ], ids=["1080P", "2K"])
    def test_12_01_resolution_gradient(self, filename, label):
        """视频分辨率梯度(1080P / 2K),用现有 fixtures → HTTP 200。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": fixture_video_base64(filename)}},
                {"type": "text", "text": "Describe this video"},
            ]}],
            "max_tokens": 2048,
        }, timeout=300)
        assert r["status"] == 200, (
            f"12_01 {label} ({filename}) expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("count", [3, 5], ids=["count=3", "count=5"])
    def test_12_02_multi_video_gradient(self, count):
        """多视频梯度(3 / 5 段叠加),用 real_2s.mp4 → 允许 200/4xx。"""
        vid = real_video_b64("real_2s.mp4")
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                *[{"type": "video_url", "video_url": {"url": vid}} for _ in range(count)],
                {"type": "text", "text": f"You see {count} short videos. Acknowledge them."},
            ]}],
            "max_tokens": 2048,
        }, timeout=300)
        assert r["status"] in (200, 400, 413, 422), (
            f"12_02 count={count} unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_12_03_video_temporal(self):
        """real_2s.mp4 + What happens in this video → content > 10 字符。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "text", "text": "What happens in this video?"},
            ]}],
            "max_tokens": 1024,
        })
        assert_oai_success(r)
        content = get_oai_content(r)
        assert len(content) > 10, (
            f"12_03 expected non-empty video description (>10 chars), got len={len(content)}: "
            f"{content!r}"
        )

    def test_12_04_image_video_combined(self):
        """真实图 sx1.jpg + 真实视频 real_2s.mp4 同一 message → content > 20。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": real_image_b64_for_video("sx1.jpg", "image/jpeg")}},
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "text", "text": "Describe both the image and the video."},
            ]}],
            "max_tokens": 1024,
        })
        assert_oai_success(r)
        content = get_oai_content(r)
        assert len(content) > 20, (
            f"12_04 expected descriptions for both image and video (>20 chars), "
            f"got len={len(content)}: {content!r}"
        )


# ============================================================
# 13 video_extension — reasoning_split 等扩展字段
# ============================================================

class TestVideoExtension:
    """扩展协议字段在视频场景的兼容性。"""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_13_01_reasoning_split(self, stream):
        """视频 + reasoning_split + thinking adaptive,流式/非流式两路,允许 200/400(扩展字段可能不支持)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": real_video_b64("real_2s.mp4")}},
                {"type": "text", "text": "Describe step by step."},
            ]}],
            "reasoning_split": True,
            "thinking": {"type": "adaptive"},
            "max_tokens": 1024,
        }, stream=stream)
        assert r["status"] in (200, 400), (
            f"13_01 stream={stream} unexpected HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )


# ============================================================
# 14 error_codes — 视频相关错误码
# ============================================================

class TestVideoErrorCodes:
    """视频相关错误码(从 TestOAIErrors 拆出)。"""

    def test_14_01_fps_out_of_range(self):
        """fps=100 显著越界 → HTTP 400(硬断言)。"""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": mp4_base64(), "fps": 100}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert_error(r, 400)
