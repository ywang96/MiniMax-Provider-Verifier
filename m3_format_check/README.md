# M3 格式校验 — 独立运行包

可独立 `pytest` 运行,覆盖 M3 模型(MiniMax-M3 等)的 OAI `/v1/chat/completions` 接口格式校验,
按 text / image / video 三个模态拆分,**collect 后共 322 个 case**(text 138 / image 99 / video 85)。

## 目录结构

```
m3_format_check/
├── conftest.py              # pytest fixture / 日志路径注入 (xdist-aware)
├── helpers.py               # HTTP 调用 + 全部断言工具 + 内嵌 base64 多媒体 dataURL
├── image_tools.py           # 任意尺寸单色 PNG 生成器 + 视频 fixture 路径
├── m3_text_tests.py         # 文本模态 138 case (基础对话/工具调用/参数压力 等)
├── m3_image_tests.py        # 图像模态 99 case  (base64/URL/多图/分辨率档位 等)
├── m3_video_tests.py        # 视频模态 85 case  (base64/URL/MOV/MKV/fps/分辨率 等)
├── pytest.ini               # python_files=m3_*_tests.py / timeout=1800s
├── requirements.txt         # httpx + pytest + pytest-timeout + pytest-xdist
├── logs/                    # 每次运行的 jsonl 日志输出目录(运行时自动生成,已 git 排除)
├── docs/                    # case 解释(中/英文双语,与 m3_*_tests.py 一一对应)
│   ├── m3_text_cases.md   / m3_text_cases_en.md    # 文本模态 case 详解
│   ├── m3_image_cases.md  / m3_image_cases_en.md   # 图像模态 case 详解
│   └── m3_video_cases.md  / m3_video_cases_en.md   # 视频模态 case 详解
└── fixtures/                # 测试用多媒体素材(总 ~107MB)
    ├── 476117246902419462.mp4    # ~5MB 卡通小马(PONY base64/URL 真实素材 case)
    ├── video_<W>x<H>.mp4         # 5 个固定分辨率(400x300/640x480/1280x720/1920x1080/3840x2160)
    ├── m3_test_videos/           # m3_video_tests.py 读这里
    │   ├── real_2s.mp4           # 105KB 真实 2 秒视频(内容理解 case)
    │   ├── 12s_real.mp4          # 628KB 12 秒视频(fps 合法档位)
    │   ├── flower-video.mp4      # ~1MB(图+视频混合 case 第二段)
    │   ├── test_video.{mov,mkv,avi}   # 03 模块真实非 MP4 格式 smoke
    │   └── D_{300,600,1200,1800}s.mp4 # 5/10/20/30min 长视频(11 模块)
    └── m3_test_images/real/      # m3_image_tests.py 读这里
        ├── sx1.jpg               # 234KB 2000×1334 海边女子
        └── zn6.jpg               # 2.2MB 4284×5712 水族馆女子(多图对比)
```

## 必填环境变量

```bash
export M3_BASE_URL="https://your-endpoint.example.com"   # API 根地址,**不带** /v1/chat/completions 后缀
export M3_API_KEY="sk-xxx"
```

## 可选环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `M3_MODEL` | `MiniMax-M3` | 主模型 ID |
| `M3_MODEL_MINI` | `MiniMax-M2-mini` | mini 模型(模型名兼容 case 用;若 endpoint 未注册同名模型该 case 会 xfail) |
| `M3_RUN_LOG` | `logs/run_<UTC-ts>.jsonl` | 自定义 jsonl 路径 |
| `M3_EXTRA_HEADERS` | (空) | JSON 串,额外注入请求头,例如 `'{"X-Custom-Header": "value"}'` |
| `M3_SKIP_REASONING_SPLIT` | `0` | `1` 时跳过依赖 `reasoning_split` 的 case(部分实现不放行该 OAI extension) |
| `M3_LONG_VIDEO_DIR` | `fixtures/m3_test_videos` | 长视频(D_*s.mp4)目录,默认走仓内 fixtures;需要换一组时改这里 |
| `M3_MEDIA_BASE_URL` | (空) | **flat** 解析器 base URL(等同 `--media-base-url`)。见下方 **URL 媒体投递** |
| `M3_GCS_BUCKET` / `M3_GCS_SIGNER_SA` / `M3_GCS_URL_DURATION` | (空) | **gcs-signed** 解析器(等同 `--gcs-bucket` / `--gcs-signer-sa` / `--gcs-url-duration`) |

## URL 媒体投递

默认套件以内联 base64(`data:` URI)发送图片/视频。开启 URL 媒体模式后,媒体改为按引用投递,
由服务端拉取而非塞进请求体。这样请求体很小(当网关对请求体解析大小有上限时很有用),同时测试
服务端的 URL 拉取路径。实现见 `url_media.py`,启动横幅会打印当前模式。

### 模式一 —— flat,预托管(`--media-base-url`)

字节与 `fixtures/` 下某文件一致的媒体改为 `<base-url>/<basename>` 发送。需自行托管文件
(套件**不上传**),按“内容哈希 → 文件名”映射。

```bash
pytest m3_image_tests.py --media-base-url=https://my-host.example.com/m3-fixtures
# 或:export M3_MEDIA_BASE_URL=https://my-host.example.com/m3-fixtures
```

只有 `fixtures/` 下存在的文件会被改写;动态生成的图片(纯色 PNG)、被填充放大的 fixture、
故意损坏的数据没有对应文件,仍走内联。

### 模式二 —— GCS 签名 URL,私有桶(`--gcs-bucket`)

将每个内联媒体上传到 `gs://<bucket>/m3/<sha256>.<ext>`(若不存在),并发送限时 V4 签名 URL。
适用于**私有**桶(无需公开访问),且覆盖**所有**媒体(含动态生成的图片,按请求内容上传)。
设置后优先于 flat 模式。

```bash
pytest m3_image_tests.py \
  --gcs-bucket=my-bucket \
  --gcs-signer-sa=signer@my-project.iam.gserviceaccount.com \
  --gcs-url-duration=12h        # 可选,默认 12h
# 或环境变量:M3_GCS_BUCKET / M3_GCS_SIGNER_SA / M3_GCS_URL_DURATION
```

前置要求:PATH 上有已认证的 `gsutil` + `gcloud`;签名 SA 需 `roles/iam.serviceAccountTokenCreator`
(用于签名)及桶对象读权限;当前账号需对该 SA 拥有 Token Creator。上传与签名 URL 按内容哈希
缓存在磁盘(`$TMPDIR/m3url_cache`),xdist 各 worker 与重复请求复用。

## 运行

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 设环境变量(见上)
export M3_BASE_URL=...
export M3_API_KEY=...

# 3a. 跑全部 322 个 case(串行,慢)
pytest

# 3b. 跑某个模态(也可拆 3 个 subprocess 并发)
pytest m3_text_tests.py
pytest m3_image_tests.py
pytest m3_video_tests.py

# 3c. pytest-xdist 并发(单 modality 内部多 worker)
pytest m3_text_tests.py -n 8

# 3d. 单个 case 复现
pytest m3_image_tests.py::TestImageBase64::test_01_01_base64_image -v
```

## 输出

- **stdout**: pytest 标准输出,每次运行开头会打印 jsonl 路径,如:
  `[m3_api_test] run log → /path/to/m3_format_check/logs/run_20260605T021800Z.jsonl`
- **jsonl 日志**: 每行一条 `oai_chat()` 调用的完整 request + response (含 chunks)
  + 状态码 + 耗时 + trace_id,失败/异常也会留 trace

## 缺失的 size-limit fixtures(按需自动下载)

服务端文件大小约束(图≤10MB / 视频≤50MB / 请求体≤64MB)的 5 个 fixture **未入包**
(总 ~150MB),`helpers.load_size_fixture()` 首次调用时会自动下载到 `fixtures/`:

| 文件 | 大小 | 远程链接 |
|------|-----:|---------|
| `image_9mb.png`  | ~9.2 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/image_9mb.png |
| `image_11mb.png` | ~11 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/image_11mb.png |
| `image_65mb.png` | ~67 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/image_65mb.png |
| `video_49mb.mp4` | ~47 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/video_49mb.mp4 |
| `video_51mb.mp4` | ~52 MB | https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test/video_51mb.mp4 |

第一次跑 `m3_image_tests.py::TestImageSizeLimit` / `m3_video_tests.py::TestVideoSizeLimit` 时
会触发自动下载并落盘到本地 `fixtures/`(后续直接读本地)。

## 集成

- 想集成进其他工程,直接 wrap `pytest` subprocess 或 `pytest.main([...])` 即可
- jsonl 格式约定见 `helpers._write_log_line` / `helpers.oai_chat` 末尾的 `record` 定义

## 总文件 ~107MB

```
107M  fixtures/  total
 60M    fixtures/m3_test_videos/         (D_*s.mp4 长视频是大头)
 38M    fixtures/video_*.mp4             (5 个分辨率档)
4.9M    fixtures/476117246902419462.mp4  (PONY)
2.4M    fixtures/m3_test_images/real/    (sx1.jpg + zn6.jpg)
```
