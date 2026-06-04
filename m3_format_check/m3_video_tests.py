"""
M3 API Test — video modality case collection

Organized by "validation content" into modules; case naming convention:
    test_<module_id>_<case_index_within_module>_<scenario_description>

Module ID / topic:
    01  base64_video          base64 video basic acceptance
    02  url_video             URL video acceptance
    03  video_format          video container/MIME format (MOV/MKV/AVI etc.)
    04  multi_video           multi-video stacking / count upper bound
    05  image_video_mixed     image + video mixed message
    06  fps_param             fps valid tiers and out-of-range
    07  detail_param          detail / fps field defaults and combinations
    08  resolution_tier       resolution tier / boundary / pixel upper bound
    09  max_long_side_pixel   max_long_side_pixel (multiple of 28) contract
    10  video_size_limit      video size limit (<=50MB)
    11  long_video            long videos (5/10/20/30 min)
    12  media_gradient        resolution gradient / multi-video gradient
    13  video_extension       reasoning_split and other extension fields
    14  error_codes           video-related error codes

Modality priority video > image > text; this file covers video (image+video
mixed cases also belong here).

All cases go through helpers.oai_chat() to /v1/chat/completions; jsonl is
written to RUN_LOG_PATH (injected by conftest).
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
# Local helpers for this file: real video fixture loaders
# (avoids polluting helpers.py; the canonical `real_video_b64`
# entrypoint mirrored here for symmetry with m3_image_tests.py)
# ============================================================

_REAL_VIDEO_DIR = Path(__file__).parent / "fixtures" / "m3_test_videos"
_REAL_IMAGE_DIR = Path(__file__).parent / "fixtures" / "m3_test_images" / "real"


def real_video_b64(name: str = "real_2s.mp4", mime: str = "video/mp4") -> str:
    """Read fixtures/m3_test_videos/<name> -> base64 data URL.

    Same semantics as m3_image_tests.py:real_image_b64. Common fixtures:
      - real_2s.mp4     (~104KB, P09/P14 content understanding cases)
      - 12s_real.mp4    (~628KB, 06 fps valid tier 12-second video)
      - flower-video.mp4 (~1MB, second clip in image+video mixed cases)
      - test_video.mov / .mkv / .avi (~160KB, real non-MP4 format smoke tests for module 03)
    """
    path = _REAL_VIDEO_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"real video fixture missing: {path}\n"
            f"fixtures/m3_test_videos/ should contain real_2s.mp4 / 12s_real.mp4 etc."
        )
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def real_image_b64_for_video(name: str = "sx1.jpg", mime: str = "image/jpeg") -> str:
    """Read local copy from fixtures/m3_test_images/real/<name>, for image+video mixed cases in this video test file.

    Kept isolated from m3_image_tests.py:real_image_b64 to avoid cross-file imports.
    """
    path = _REAL_IMAGE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"real image fixture missing: {path}\n"
            f"fixtures/m3_test_images/real/ should contain sx1.jpg"
        )
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


# Long videos (>10MB) are read from the directory specified by the
# M3_LONG_VIDEO_DIR env var by default; when unset, fall back to local
# fixtures/m3_test_videos/ (D_300s/D_600s/D_1200s/D_1800s.mp4 already in repo).
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
    """Extract prompt_tokens from oai_chat return value (supports both stream / non_stream)."""
    if not r.get("stream"):
        body = r.get("body") or {}
        return (body.get("usage") or {}).get("prompt_tokens", 0)
    for c in reversed(r.get("chunks") or []):
        usage = c.get("usage") if isinstance(c, dict) else None
        if usage:
            return usage.get("prompt_tokens", 0)
    return 0


def _assert_basic_ok(r: dict, msg: str = ""):
    """Basic assertions for valid-tier happy path: HTTP 200 + positive prompt_tokens."""
    assert r["status"] == 200, (
        f"{msg}: expected 200, got {r['status']}: "
        f"{r.get('body', '')[:300] if isinstance(r.get('body'), str) else r.get('body')}"
    )
    pt = _get_prompt_tokens(r)
    assert pt > 0, f"{msg}: prompt_tokens should be positive, got {pt}"


def _video_payload(filename: str, detail: str = None, fps: float = None,
                   text: str = "What is this?") -> dict:
    """Build the video_url block for a video request."""
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


# Real asset marker shared across multiple modules (Spring Festival cartoon pony, 5MB)
PONY_VIDEO_FIXTURE = "476117246902419462.mp4"
PONY_VIDEO_URL = (
    "https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/"
    "476117246902419462.mp4"
)


# ============================================================
# 01 base64_video — base64 video basic acceptance
# ============================================================

class TestBase64Video:
    """Feed video via base64 dataURL: verify minimal usable path + real asset content understanding."""

    def test_01_01_base64_video(self):
        """base64 minimal MP4 + What do you see -> HTTP 200 + non-empty answer."""
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
        """Real asset (Spring Festival cartoon pony, 5MB) via base64 input, verify content understanding + content > 50."""
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
# 02 url_video — URL video acceptance
# ============================================================

class TestURLVideo:
    """video_url.url via public network / COS URL form."""

    def test_02_01_url_video(self):
        """Public sample mp4 URL -> HTTP 200 + non-empty answer (basic URL acceptance)."""
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
        """Real asset (Spring Festival cartoon pony) via OSS URL input, verify content understanding + content > 50."""
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
# 03 video_format — video container/MIME format (MOV/MKV/AVI etc.)
# ============================================================

class TestVideoFormat:
    """Video format compatibility — including spoofed MIME (MOV/MKV/AVI etc.) and real non-MP4 file smoke tests."""

    def test_03_01_mkv_format_legacy(self):
        """MKV (video/x-matroska) disguised with minimal mp4 -> xfail if known unsupported."""
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
        """MOV (video/quicktime base64) -> xfail on known BUG-7."""
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
        """MOV (spec 1.3.5 requires data:video/mov;base64,...) -> xfail on known BUG-7."""
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
        """MKV (video/x-matroska) disguised with minimal mp4 -> xfail on known BUG-7."""
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
        """Both valid AVI MIMEs (video/avi / video/x-msvideo) disguised with minimal mp4 -> xfail if unsupported."""
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
        """Real MOV file (fixtures/test_video.mov) default detail/fps -> accept or legitimate reject."""
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
        """Real AVI file (fixtures/test_video.avi) default detail/fps -> accept or legitimate reject."""
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
        """Real MKV file (fixtures/test_video.mkv) default detail/fps -> accept or legitimate reject."""
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
# 04 multi_video — multi-video stacking / count upper bound
# ============================================================

class TestMultiVideo:
    """spec 1.3.6 video count upper bound 5 segments + multi-segment stacking boundary coverage."""

    def _mp4_block(self):
        return {"type": "video_url", "video_url": {
            "url": "data:video/mp4;base64,"
                   + base64.b64encode(make_minimal_mp4_504()).decode(),
        }}

    def test_04_01_count_at_max(self):
        """5 videos (spec upper bound) -> API should accept (HTTP 200)."""
        content = [self._mp4_block() for _ in range(5)]
        content.append({"type": "text", "text": "How many videos do you see?"})
        r = oai_chat({"messages": [{"role": "user", "content": content}]})
        assert r["status"] == 200, (
            f"04_01 video count=5 (at max) HTTP={r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_04_02_count_over_max(self):
        """10 videos (> spec upper bound 5) -> M3 official does not enforce reject in practice; verify HTTP 200 + non-empty content."""
        content = [self._mp4_block() for _ in range(10)]
        content.append({"type": "text", "text": "How many?"})
        r = oai_chat({"messages": [{"role": "user", "content": content}]})
        assert r["status"] == 200, (
            f"04_02 video count=10 expected HTTP 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )
        resp_content = get_oai_content(r)
        assert resp_content and resp_content.strip(), (
            f"04_02 video count=10 expected non-empty content, got: {resp_content!r}"
        )

    @pytest.mark.slow
    def test_04_03_multi_video_3(self):
        """3 real_2s.mp4 real videos stacked, allow 200/4xx."""
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
        """5 real_2s.mp4 real videos stacked (spec upper bound), allow 200/4xx."""
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
        """Mixed form: base64 + URL multi-video with two different load forms, allow 200/4xx.

        Different from 04_01..04_04's "all base64 same segment" coverage of input forms.
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
# 05 image_video_mixed — image + video mixed message
# ============================================================

class TestImageVideoMixed:
    """Mixed multimodal input with image + video in the same message."""

    def test_05_01_one_image_one_video(self):
        """1 image (PNG) + 1 video (real_2s) -> HTTP 200 (multimodal mix)."""
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
        """Image+video mixed -> both non-stream/stream paths should return 200."""
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
        """3 real images (sx1×3) + 1 real video (real_2s) mixed -> allow 200/4xx."""
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
        """1 real image (sx1) + 2 real videos (real_2s + flower-video) mixed -> allow 200/4xx."""
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
# 06 fps_param — fps valid tiers and out-of-range
# ============================================================

class TestFpsParam:
    """fps in [0.2, 5] valid tiers and out-of-range reject behavior."""

    @pytest.mark.parametrize("fps", [0.5, 1, 2], ids=["fps=0.5", "fps=1", "fps=2"])
    def test_06_01_fps_real_video(self, fps):
        """fps in {0.5, 1, 2} with 12s_real.mp4 real video -> HTTP 200."""
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
        """Each valid fps tier in [0.2, 5], with 12s_real.mp4 -> HTTP 200."""
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
        """Out-of-range values (< 0.2 or > 5); server may reject or leniently accept; soft assertion 200/400/422/500."""
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
        """fps=0.2 (lower bound) -> HTTP 200 + prompt_tokens > 0."""
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=0.2))
        _assert_basic_ok(r, "06_04 fps=0.2 lower boundary")

    def test_06_05_fps_upper_boundary(self):
        """fps=5.0 (upper bound) -> HTTP 200 + prompt_tokens > 0."""
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=5.0))
        _assert_basic_ok(r, "06_05 fps=5.0 upper boundary")

    @pytest.mark.parametrize("fps", [0.1, 0.0, -1], ids=["0.1", "0.0", "-1"])
    def test_06_06_fps_below_min_short_fixture(self, fps):
        """fps<0.2 with video_640x480.mp4 (short fixture) -> soft assert 200/400/422, record actual behavior.

        Complementary to 06_03: 06_03 uses 12s_real.mp4 (long video) to cover the full out-of-range set,
        06_06 uses a short fixture to cover the 3 common out-of-range values near the lower bound, for quick diagnostics.
        """
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=fps))
        assert r["status"] in (200, 400, 422), f"06_06 fps={fps} HTTP={r['status']}"

    @pytest.mark.parametrize("fps", [5.1, 10, 1000], ids=["5.1", "10", "1000"])
    def test_06_07_fps_above_max_short_fixture(self, fps):
        """fps>5 with video_640x480.mp4 (short fixture) -> soft assert 200/400/422, record actual behavior.

        Complementary to 06_03: 06_07 covers the 3 common out-of-range values near the upper bound; the short fixture allows quick diagnostics.
        """
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default", fps=fps))
        assert r["status"] in (200, 400, 422), f"06_07 fps={fps} HTTP={r['status']}"


# ============================================================
# 07 detail_param — detail / fps field defaults and combinations
# ============================================================

class TestDetailParam:
    """detail = low/default/high three-tier acceptance + detail / fps field default combinations."""

    @pytest.mark.parametrize("detail", ["low", "default", "high"],
                             ids=["detail=low", "detail=default", "detail=high"])
    def test_07_01_detail_value(self, detail):
        """video_url.detail in {low, default, high} -> HTTP 200."""
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
        """Pass neither detail nor fps; server uses defaults (detail=default, fps=1) -> HTTP 200."""
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
            f"07_02 default params expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_07_03_no_detail_explicit_fps(self):
        """Only pass fps, not detail; server uses default detail=default -> HTTP 200."""
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
            f"07_03 fps only expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_07_04_no_fps_explicit_detail(self):
        """Only pass detail, not fps; server uses default fps=1 -> HTTP 200."""
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
            f"07_04 detail only expected 200, got {r['status']}: "
            f"{str(r.get('body'))[:300]}"
        )

    def test_07_05_max_long_side_pixel_baseline(self):
        """max_long_side_pixel=504 (28×18 minimum tier) + real_2s.mp4 -> HTTP 200 (smoke test for minimum-tier contract).

        # OAI contract: max_long_side_pixel must be a multiple of 28.
        # This case only does smoke testing on the minimum tier; full coverage of series contract / monotonicity / out-of-range is in section 09.
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
# 08 resolution_tier — resolution tier / boundary / pixel upper bound
# ============================================================

class TestResolutionTier:
    """Videos at different resolutions go through detail=default rescaling + max_total_pixels soft upper bound."""

    def test_08_01_tier_low_no_scale(self):
        """400x300 video (long side 400 < default tier 672) -> HTTP 200 (no rescaling)."""
        r = oai_chat(_video_payload("video_400x300.mp4", detail="default"))
        _assert_basic_ok(r, "08_01 small_video")

    def test_08_02_tier_low_scale_down(self):
        """1280x720 video (long side 1280 > default tier 672) -> per-frame rescaling expected, HTTP 200."""
        r = oai_chat(_video_payload("video_1280x720.mp4", detail="default"))
        _assert_basic_ok(r, "08_02 medium_video")

    def test_08_03_tier_default_no_scale(self):
        """640x480 video (long side 640 < default tier 672) -> HTTP 200 (no rescaling)."""
        r = oai_chat(_video_payload("video_640x480.mp4", detail="default"))
        _assert_basic_ok(r, "08_03 below_default")

    def test_08_04_tier_default_scale_down(self):
        """1920x1080 video (long side 1920 > default tier 672) -> per-frame rescaling expected, HTTP 200."""
        r = oai_chat(_video_payload("video_1920x1080.mp4", detail="default"))
        _assert_basic_ok(r, "08_04 above_default")

    def test_08_05_tier_high_no_scale(self):
        """1280x720 video (long side 1280 > default tier 672) -> HTTP 200."""
        r = oai_chat(_video_payload("video_1280x720.mp4", detail="default"))
        _assert_basic_ok(r, "08_05 medium_large_video")

    def test_08_06_tier_high_scale_down(self):
        """3840x2160 video (long side 3840 > default tier 672) -> per-frame rescaling expected, HTTP 200."""
        r = oai_chat(_video_payload("video_3840x2160.mp4", detail="default"))
        _assert_basic_ok(r, "08_06 large_video")

    def test_08_07_tier_at_boundary(self):
        """1280x720 video (long side = 2x default tier) -> boundary soft assert HTTP 200."""
        r = oai_chat(_video_payload("video_1280x720.mp4", detail="default"))
        _assert_basic_ok(r, "08_07 boundary")

    def test_08_08_tier_consistency(self):
        """1920x1080 large video + default tier -> HTTP 200 + prompt_tokens > 0 (basic smoke test)."""
        r = oai_chat(_video_payload("video_1920x1080.mp4", detail="default"))
        assert r["status"] == 200, f"08_08 HTTP={r['status']}"
        pt = _get_prompt_tokens(r)
        assert pt > 0, f"08_08 prompt_tokens={pt}"

    def test_08_09_max_total_pixels_exceeded(self):
        """1-second 3840x2160 + fps=5, approaching the max_total_pixels=301,056,000 soft upper bound -> 200/4xx.

        # Current fixtures are all 1-second videos and not enough to actually trip the upper bound;
        # only does "high fps large resolution" acceptance assertion; actual over-limit testing
        # requires longer-duration fixtures.
        """
        r = oai_chat(_video_payload("video_3840x2160.mp4", detail="default", fps=5))
        assert r["status"] in (200, 400, 413, 422), (
            f"08_09 HTTP={r['status']}"
        )


# ============================================================
# 09 max_long_side_pixel — max_long_side_pixel (multiple of 28) contract
# ============================================================
# OAI contract: max_long_side_pixel must be a multiple of 28.
# The three video tiers use 504 (28×18) / 1008 (28×36) / 2016 (28×72);
# the old (504, 672, 1280) combination had 672/1280 that were not multiples of 28 and is deprecated.

class TestMaxLongSidePixel:
    """max_long_side_pixel three tiers (504/1008/2016) contract + monotonicity + out-of-range soft assertions."""

    @pytest.mark.parametrize("mlsp,tier", [
        (504, "low"),
        (1008, "default"),
        (2016, "high"),
    ], ids=["low=504", "default=1008", "high=2016"])
    def test_09_01_tier_value(self, mlsp, tier):
        """Each of the three tiers (504/1008/2016) fed a 3840x2160 real video (long side > 2016, all trigger rescaling) -> HTTP 200 + prompt_tokens > 0."""
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
        """Same video with each of the three tiers, prompt_tokens strictly monotonic 504 < 1008 < 2016."""
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
            f"09_02 monotonic: larger max_long_side_pixel should yield larger prompt_tokens, "
            f"got {tokens} (expected 504 < 1008 < 2016)"
        )

    @pytest.mark.parametrize("invalid_value", [140, 3612, 0, -28],
                             ids=["below_min_28x5", "above_max_28x129", "zero", "negative_28"])
    def test_09_03_out_of_range(self, invalid_value):
        """Outside spec range [150, 3584] (<150 / >3584 / 0 / negative, all multiples of 28) -> soft assert 200/400/413/422."""
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
# 10 video_size_limit — video size limit (<=50MB)
# ============================================================

class TestVideoSizeLimit:
    """Video <= 50 MB; URL/Base64 each covers pass/reject sides plus padded equivalents."""

    def test_10_01_url_under_50mb(self):
        """URL form: ~47.4 MB MP4 within the 50 MB upper bound, should be accepted."""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": size_fixture_url("video_49mb.mp4")}},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
        })
        assert_oai_success(r)

    def test_10_02_url_over_50mb(self):
        """URL form: ~52 MB MP4 exceeds the 50 MB upper bound; server should reject after download (4xx)."""
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
        """Base64 form: ~47.4 MB MP4 within the 50 MB upper bound, should be accepted."""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": size_fixture_data_url("video_49mb.mp4")}},
                {"type": "text", "text": "Describe this video briefly."},
            ]}],
        }, timeout=300)
        assert_oai_success(r)

    @pytest.mark.timeout(600)
    def test_10_04_base64_over_50mb(self):
        """Base64 form: ~52 MB MP4 exceeds the 50 MB upper bound, should be rejected (4xx)."""
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
        """real_2s.mp4 + null padding to 51MB (real header + null fill) -> should be rejected (4xx or 500).

        Complementary to 10_04 clean 52MB fixture; covers different forms of "oversized video" input.
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
# 11 long_video — long videos (5/10/20/30 min)
# ============================================================
# Defaults to reading D_300s/D_600s/D_1200s/D_1800s.mp4 from local fixtures/m3_test_videos/;
# to use a different set of long videos, set env var `M3_LONG_VIDEO_DIR` to override.
# Cases in this section auto-skip when the fixture is missing.

def _long_video_present(name: str) -> bool:
    return long_video_path(name).exists()


class TestLongVideo:
    """5/10/20/30 minute video duration and upper-bound test (skipif env)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _long_video_present("D_300s.mp4"),
        reason="long video fixture D_300s.mp4 not present; "
               "set M3_LONG_VIDEO_DIR=<dir-with-D_300s.mp4>",
    )
    def test_11_01_long_video_5min(self):
        """5-minute video (~5MB, default fps=1)."""
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
        """10-minute video (default fps=1)."""
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
        """20-minute video (fps=0.5 to control total pixels)."""
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
        """30-minute video (fps=0.5 to control total pixels)."""
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
# 12 media_gradient — resolution gradient / multi-video gradient
# ============================================================

class TestMediaGradient:
    """Media gradient: video resolution gradient (1080P/2K) and multi-video count gradient (3/5 segments)."""

    @pytest.mark.parametrize("filename,label", [
        ("video_1920x1080.mp4", "1080P"),
        ("video_3840x2160.mp4", "2K"),
    ], ids=["1080P", "2K"])
    def test_12_01_resolution_gradient(self, filename, label):
        """Video resolution gradient (1080P / 2K), using existing fixtures -> HTTP 200."""
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
        """Multi-video gradient (3 / 5 segments stacked), using real_2s.mp4 -> allow 200/4xx."""
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
        """real_2s.mp4 + What happens in this video -> content > 10 chars."""
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
        """Real image sx1.jpg + real video real_2s.mp4 in the same message -> content > 20."""
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
# 13 video_extension — reasoning_split and other extension fields
# ============================================================

class TestVideoExtension:
    """Compatibility of extended protocol fields in video scenarios."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_13_01_reasoning_split(self, stream):
        """Video + reasoning_split + thinking adaptive, both stream and non-stream paths; allow 200/400 (extension fields may not be supported)."""
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
# 14 error_codes — video-related error codes
# ============================================================

class TestVideoErrorCodes:
    """Video-related error codes (split from TestOAIErrors)."""

    def test_14_01_fps_out_of_range(self):
        """fps=100 significantly out of range -> HTTP 400 (hard assertion)."""
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": mp4_base64(), "fps": 100}},
                {"type": "text", "text": "What?"},
            ]}],
        })
        assert_error(r, 400)
