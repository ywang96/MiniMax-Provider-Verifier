"""
M3 测试用助手:任意尺寸单色 PNG 字节流生成器 + 预置视频 fixture 路径

设计目标:
- 零三方依赖(不引入 Pillow),与 helpers.py:make_png_1x1 风格一致
- 单色 PNG 生成,IDAT 用 zlib 压缩,4000x3000 PNG 实际只有几 KB
- 视频 fixtures 走预置文件路径,fixtures/ 下放固定分辨率 mp4

helpers.py 不动,本文件独立,供 test_resolution_tier.py 引用。
"""
import base64
import struct
import zlib
from pathlib import Path
from typing import Optional


# ---------- PNG 生成器 ----------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """生成单个 PNG chunk(参考 helpers.make_png_1x1 的 _chunk 写法)"""
    payload = chunk_type + data
    crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    return struct.pack(">I", len(data)) + payload + crc


def make_png_bytes(width: int, height: int, r: int = 255, g: int = 0, b: int = 0) -> bytes:
    """
    生成 width x height 像素的单色 RGB PNG 字节流。

    实现要点:
    - color type = 2 (RGB,与 make_png_1x1 一致)
    - bit depth = 8
    - 每行前一个 filter byte (0x00 = None),后跟 width 个 RGB 三元组
    - 整张 raw 数据走 zlib.compress;单色图压缩比极高,4000x3000 PNG 约 8-12KB
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions: {width}x{height}")

    header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    pixel = bytes([r, g, b])
    row = b"\x00" + pixel * width  # filter byte + RGB pixels
    raw = row * height
    idat = zlib.compress(raw, level=9)

    return header + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def make_png_base64(width: int, height: int, r: int = 255, g: int = 0, b: int = 0) -> str:
    """make_png_bytes 的 data URL 包装"""
    return "data:image/png;base64," + base64.b64encode(make_png_bytes(width, height, r, g, b)).decode()


# ---------- 视频 fixtures ----------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def fixture_path(name: str) -> Path:
    """fixtures/ 下文件的绝对路径"""
    return FIXTURES_DIR / name


def fixture_video_base64(name: str, mime: str = "video/mp4") -> str:
    """读取 fixtures/ 下视频文件并返回 data URL 形式的 base64"""
    p = fixture_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"fixture video not found: {p}\n"
            f"请先生成 fixture(见 data/m3_api_test/fixtures/README.md)"
        )
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# 预置的几个固定分辨率 fixture(详见 fixtures/README.md)
# 文件名 → (width, height, 用途说明)
VIDEO_FIXTURES = {
    "video_400x300.mp4":   (400, 300,   "low 档不缩放场景"),
    "video_640x480.mp4":   (640, 480,   "default 档不缩放场景"),
    "video_1280x720.mp4":  (1280, 720,  "high 档不缩放/low 档缩放场景"),
    "video_1920x1080.mp4": (1920, 1080, "default 档缩放场景"),
    "video_3840x2160.mp4": (3840, 2160, "high 档缩放场景"),
}


# COS 备份链接(详见 fixtures/README.md);test_resolution_tier.py 的 URL 系列 case 走这条
COS_VIDEO_BASE = "https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test"


def fixture_video_url(name: str) -> str:
    """返回 fixtures/ 下视频文件对应的 COS 直链(用于 video_url.url 直接传 URL 形式)"""
    if name not in VIDEO_FIXTURES:
        raise KeyError(f"unknown fixture: {name}; known: {list(VIDEO_FIXTURES)}")
    return f"{COS_VIDEO_BASE}/{name}"
