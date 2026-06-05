"""URL media delivery for the m3_format_check suite.

By default the suite sends image/video as inline base64 (`data:` URIs). When URL
media mode is enabled, each inline asset is replaced with a URL before the
request is sent, so media is fetched server-side instead of embedded in the
request body. This keeps request bodies tiny (useful when a gateway caps the
request-body parse size) and exercises the server's URL-fetch path.

Two resolvers (selected in conftest from CLI options / env vars):

- **flat** (`--media-base-url`): map an asset whose bytes match a file under
  `fixtures/` to `<base-url>/<basename>`. You pre-host the files; the suite does
  not upload. Mapping is by content hash → basename.

- **gcs-signed** (`--gcs-bucket` + `--gcs-signer-sa`): upload each asset to
  `gs://<bucket>/m3/<sha256>.<ext>` (if absent) and emit a time-limited V4
  signed URL. Works with a private bucket — no public access needed. Requires
  `gsutil` + `gcloud` on PATH and a signer service account with Token Creator
  (for signing) and object read on the bucket.

Both resolvers leave media inline when they cannot produce a URL (e.g. a flat
miss, or a decode failure), so the request still goes through.
"""
import base64
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
    "image/gif": "gif", "image/webp": "webp", "image/bmp": "bmp",
    "video/mp4": "mp4", "video/quicktime": "mov", "video/mov": "mov",
    "video/x-matroska": "mkv", "video/avi": "avi", "video/x-msvideo": "avi",
}


def _decode_data_uri(uri: str):
    """data:<mime>[;params];base64,<data> -> (mime, raw bytes) or (None, None)."""
    try:
        head, b64 = uri.split(",", 1)
    except ValueError:
        return None, None
    mime = (head[5:].split(";")[0] or "application/octet-stream").lower()
    s = b64.strip()
    s += "=" * (-len(s) % 4)  # tolerate missing padding
    try:
        return mime, base64.b64decode(s)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# Resolvers: (raw_bytes, mime) -> url or None
# --------------------------------------------------------------------------- #
class FlatResolver:
    """Map fixtures to <base_url>/<basename> by content hash (no upload)."""

    def __init__(self, base_url: str, manifest: dict):
        self.base = base_url.rstrip("/")
        self.manifest = manifest

    def resolve(self, raw: bytes, mime: str):
        name = self.manifest.get(hashlib.sha256(raw).hexdigest())
        return f"{self.base}/{name}" if name else None


class GcsSignedResolver:
    """Upload to a (private) GCS bucket and return a V4 signed URL per object."""

    def __init__(self, bucket: str, signer_sa: str, duration: str = "12h",
                 region: str = "us", cache_dir: str = None):
        self.bucket = bucket.replace("gs://", "").rstrip("/")
        self.sa = signer_sa
        self.duration = duration
        self.region = region
        self.cache_dir = cache_dir or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "m3url_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._mem = {}

    def resolve(self, raw: bytes, mime: str):
        ext = _EXT.get(mime.lower(), "bin")
        key = f"{hashlib.sha256(raw).hexdigest()[:32]}.{ext}"
        cache_file = os.path.join(self.cache_dir, key + ".url")
        with self._lock:
            if key in self._mem:
                return self._mem[key]
            if os.path.exists(cache_file):
                url = open(cache_file).read().strip()
                if url:
                    self._mem[key] = url
                    return url
            blob = os.path.join(self.cache_dir, key)
            if not os.path.exists(blob):
                with open(blob, "wb") as f:
                    f.write(raw)
            subprocess.run(
                ["gsutil", "-q", "cp", blob, f"gs://{self.bucket}/m3/{key}"],
                check=True)
            out = subprocess.run(
                ["gcloud", "storage", "sign-url", f"gs://{self.bucket}/m3/{key}",
                 "--impersonate-service-account", self.sa,
                 f"--duration={self.duration}", f"--region={self.region}",
                 "--format=json"],
                capture_output=True, text=True, check=True)
            d = json.loads(out.stdout)
            url = (d[0] if isinstance(d, list) else d)["signed_url"]
            with open(cache_file, "w") as f:
                f.write(url)
            self._mem[key] = url
            return url


# --------------------------------------------------------------------------- #
def build_manifest(fixtures_dir) -> dict:
    """Map sha256(file bytes) -> basename for every file under fixtures_dir."""
    manifest = {}
    root = Path(fixtures_dir)
    if not root.exists():
        return manifest
    for p in root.rglob("*"):
        if p.is_file():
            try:
                manifest[hashlib.sha256(p.read_bytes()).hexdigest()] = p.name
            except OSError:
                pass
    return manifest


def rewrite_payload(payload: dict, resolver) -> int:
    """Rewrite inline data: media to URLs using `resolver`. Returns count rewritten."""
    rewritten = 0
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("image_url", "video_url"):
                ref = part.get(key)
                if not (isinstance(ref, dict) and isinstance(ref.get("url"), str)):
                    continue
                if not ref["url"].startswith("data:"):
                    continue
                mime, raw = _decode_data_uri(ref["url"])
                if raw is None:
                    continue
                try:
                    url = resolver.resolve(raw, mime)
                except Exception:
                    url = None  # fall back to inline on resolver error
                if url:
                    ref["url"] = url
                    rewritten += 1
    return rewritten
