"""URL media delivery for the m3_format_check suite.

By default the suite sends image/video as inline base64 (`data:` URIs). When a
base URL is supplied (via `--media-base-url` or `M3_MEDIA_BASE_URL`), each inline
asset whose bytes match a file under `fixtures/` is replaced with
`<base-url>/<filename>` before the request is sent, so media is fetched
server-side from a pre-hosted location instead of embedded in the request body.

This keeps request bodies tiny (e.g. to stay under a gateway's body-parse limit)
and exercises the server's URL-fetch path.

Pre-hosting contract:
  Host every file you care about at `<base-url>/<basename>`. The mapping is by
  *content hash* → basename, so only fixtures present on disk are rewritten.
  Dynamically generated images (solid-color PNGs, padded/oversized fixtures,
  intentionally-corrupted blobs) have no matching file and are left inline.
"""
import base64
import hashlib
from pathlib import Path


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


def _decode_data_uri(uri: str):
    """data:<mime>[;params];base64,<data> -> raw bytes, or None if undecodable."""
    try:
        _, b64 = uri.split(",", 1)
    except ValueError:
        return None
    s = b64.strip()
    s += "=" * (-len(s) % 4)  # tolerate missing padding
    try:
        return base64.b64decode(s)
    except Exception:
        return None


def rewrite_payload(payload: dict, base_url: str, manifest: dict) -> int:
    """Rewrite inline data: media to `<base_url>/<basename>` for known fixtures.

    Returns the number of parts rewritten (for logging/diagnostics).
    """
    base = base_url.rstrip("/")
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
                raw = _decode_data_uri(ref["url"])
                if raw is None:
                    continue
                name = manifest.get(hashlib.sha256(raw).hexdigest())
                if name:
                    ref["url"] = f"{base}/{name}"
                    rewritten += 1
    return rewritten
