# M3 Format Validation — Standalone Test Package

A self-contained `pytest` package that validates the OAI `/v1/chat/completions`
contract for M3-family models (MiniMax-M3, etc.), split across text / image /
video modalities. **322 cases collected** (text 138 / image 99 / video 85).

## Directory layout

```
m3_format_check/
├── conftest.py              # pytest fixtures + xdist-aware log path injection
├── helpers.py               # HTTP client + all assertion helpers + embedded base64 media data URLs
├── image_tools.py           # Arbitrary-size solid-color PNG generators + video fixture paths
├── m3_text_tests.py         # Text modality, 138 cases (basic chat, tool calls, param stress, ...)
├── m3_image_tests.py        # Image modality, 99 cases (base64, URL, multi-image, resolution tiers, ...)
├── m3_video_tests.py        # Video modality, 85 cases (base64, URL, MOV/MKV, fps, resolution, ...)
├── pytest.ini               # python_files=m3_*_tests.py / timeout=1800s
├── requirements.txt         # httpx + pytest + pytest-timeout + pytest-xdist
├── logs/                    # Per-run jsonl output (auto-created, git-ignored)
├── docs/                    # Per-case explanations (bilingual zh/en, one-to-one with m3_*_tests.py)
│   ├── m3_text_cases.md   / m3_text_cases_en.md    # Text modality case reference
│   ├── m3_image_cases.md  / m3_image_cases_en.md   # Image modality case reference
│   └── m3_video_cases.md  / m3_video_cases_en.md   # Video modality case reference
└── fixtures/                # Test media assets (~107MB total)
    ├── 476117246902419462.mp4    # ~5MB cartoon pony clip (PONY base64/URL real-asset cases)
    ├── video_<W>x<H>.mp4         # 5 fixed resolutions (400x300/640x480/1280x720/1920x1080/3840x2160)
    ├── m3_test_videos/           # Read by m3_video_tests.py
    │   ├── real_2s.mp4           # 105KB real 2-second video (content understanding cases)
    │   ├── 12s_real.mp4          # 628KB 12-second video (fps valid-tier cases)
    │   ├── flower-video.mp4      # ~1MB (second clip in image+video mixed cases)
    │   ├── test_video.{mov,mkv,avi}   # Real non-MP4 formats for module 03 smoke
    │   └── D_{300,600,1200,1800}s.mp4 # 5/10/20/30-min long videos (module 11)
    └── m3_test_images/real/      # Read by m3_image_tests.py
        ├── sx1.jpg               # 234KB 2000×1334, woman by the sea
        └── zn6.jpg               # 2.2MB 4284×5712, woman at aquarium (multi-image comparison)
```

## Required environment variables

```bash
export M3_BASE_URL="https://your-endpoint.example.com"   # API base URL, **without** the /v1/chat/completions suffix
export M3_API_KEY="sk-xxx"
```

## Optional environment variables

| Variable | Default | Description |
|------|------|------|
| `M3_MODEL` | `MiniMax-M3` | Primary model ID |
| `M3_MODEL_MINI` | `MiniMax-M2-mini` | Mini model ID (used by the model-name compatibility case; if no such model is registered on the endpoint, that case is xfailed) |
| `M3_RUN_LOG` | `logs/run_<UTC-ts>.jsonl` | Override the jsonl output path |
| `M3_EXTRA_HEADERS` | (empty) | JSON string of extra request headers to inject, e.g. `'{"X-Custom-Header": "value"}'` |
| `M3_SKIP_REASONING_SPLIT` | `0` | Set to `1` to skip cases that depend on `reasoning_split` (some implementations do not pass through this OAI extension) |
| `M3_LONG_VIDEO_DIR` | `fixtures/m3_test_videos` | Long-video (D_*s.mp4) directory. Defaults to the in-repo fixtures; override to point at a different set. |
| `M3_MEDIA_BASE_URL` | (empty) | Serve fixtures from this base URL instead of inline base64 (same as the `--media-base-url` option). See **URL media delivery** below. |

## URL media delivery (`--media-base-url`)

By default the suite embeds image/video as inline base64 (`data:` URIs). Pass a
base URL to instead deliver media by reference — each asset whose bytes match a
file under `fixtures/` is sent as `<base-url>/<basename>`, so the server fetches
it instead of receiving it in the request body. This keeps request bodies tiny
(useful when a gateway caps the request-body parse size) and exercises the
server's URL-fetch path.

```bash
# via CLI option (takes precedence)
pytest m3_image_tests.py --media-base-url=https://my-host.example.com/m3-fixtures
# or via env var
export M3_MEDIA_BASE_URL=https://my-host.example.com/m3-fixtures
pytest m3_image_tests.py
```

**Pre-hosting contract:** you must host the fixtures yourself at
`<base-url>/<basename>` (the suite does **not** upload). Mapping is by content
hash → basename, so only files present under `fixtures/` are rewritten.
Dynamically generated images (solid-color PNGs), padded/oversized fixtures, and
intentionally-corrupted blobs have no matching file and stay inline. The startup
banner prints `media URL mode → <base-url> (<N> fixtures mapped)`. Implementation
in `url_media.py`.

## Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (see above)
export M3_BASE_URL=...
export M3_API_KEY=...

# 3a. Run all 322 cases (serial, slow)
pytest

# 3b. Run a single modality (or fan out as 3 parallel subprocesses)
pytest m3_text_tests.py
pytest m3_image_tests.py
pytest m3_video_tests.py

# 3c. Parallelize within a modality via pytest-xdist
pytest m3_text_tests.py -n 8

# 3d. Reproduce a single case
pytest m3_image_tests.py::TestImageBase64::test_01_01_base64_image -v
```

## Output

- **stdout**: standard pytest output. The first line shows the jsonl path, e.g.
  `[m3_api_test] run log → /path/to/m3_format_check/logs/run_20260605T021800Z.jsonl`
- **jsonl log**: one line per `oai_chat()` call, containing the full request + full
  response (or per-chunk stream array) + status code + elapsed ms + trace_id.
  Failures and exceptions also leave a trace line.

## Missing size-limit fixtures (auto-downloaded on demand)

The 5 fixtures used for server-side size-limit cases (image ≤ 10MB / video ≤ 50MB
/ request body ≤ 64MB) are **not vendored** (~150MB total).
`helpers.load_size_fixture()` downloads them to `fixtures/` on first access:

| File | Size | Remote URL |
|------|-----:|---------|
| `image_9mb.png`  | ~9.2 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/image_9mb.png |
| `image_11mb.png` | ~11 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/image_11mb.png |
| `image_65mb.png` | ~67 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/image_65mb.png |
| `video_49mb.mp4` | ~47 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/video_49mb.mp4 |
| `video_51mb.mp4` | ~52 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/video_51mb.mp4 |

The first run of `m3_image_tests.py::TestImageSizeLimit` /
`m3_video_tests.py::TestVideoSizeLimit` triggers the download and caches it
under `fixtures/` (subsequent runs read locally).

## Integration

- To embed this suite into another project, wrap `pytest` as a subprocess or
  call `pytest.main([...])` directly.
- The jsonl schema is defined inline in `helpers._write_log_line` /
  `helpers.oai_chat` (see the `record` dict at the end of `oai_chat`).

## Total footprint ~107MB

```
107M  fixtures/  total
 60M    fixtures/m3_test_videos/         (D_*s.mp4 long videos dominate)
 38M    fixtures/video_*.mp4             (5 resolution tiers)
4.9M    fixtures/476117246902419462.mp4  (PONY)
2.4M    fixtures/m3_test_images/real/    (sx1.jpg + zn6.jpg)
```
