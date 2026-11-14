"""Download the advanced document and verify its published SHA-256."""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "advanced" / "source"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    target = ROOT / manifest["source_file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"Downloading {manifest['url']} ...")
        request = urllib.request.Request(
            manifest["url"], headers={"User-Agent": "day7-advanced-rag/1.0"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            target.write_bytes(response.read())
    actual = sha256(target)
    if actual != manifest["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: expected {manifest['sha256']}, got {actual}")
    print(f"OK {target} ({target.stat().st_size} bytes, sha256={actual})")


if __name__ == "__main__":
    main()
