#!/usr/bin/env python3
"""Public-repo safety guard: blocks secrets, raw data files, and large blobs before commit."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB hard cap for any tracked file

BLOCKED_EXTENSIONS = {
    ".env",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".csv",
    ".tsv",
    ".parquet",
    ".geoparquet",
    ".duckdb",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pmtiles",
    ".mbtiles",
    ".gpkg",
    ".geojson",
    ".shp",
    ".dbf",
    ".tif",
    ".tiff",
    ".zip",
    ".gz",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".pkl",
    ".joblib",
}

BLOCKED_BASENAMES = {
    ".env",
    "credentials.json",
    "service_account.json",
}

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub Token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
    ("Google / Gemini API Key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("Private Key Header", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "Hardcoded API Secret Assignment",
        re.compile(
            r"""(?i)(api_key|secret_key|access_token|password)\s*=\s*['"][A-Za-z0-9_\-]{16,}['"]"""
        ),
    ),
]


def get_staged_files() -> list[Path]:
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line.strip()) for line in res.stdout.splitlines() if line.strip()]


def main() -> int:
    files = get_staged_files()
    if not files:
        print("[safety] No staged files to check.")
        return 0

    violations: list[str] = []

    for path in files:
        if not path.exists():
            continue

        # Block anything under data/
        if path.parts and path.parts[0] == "data":
            violations.append(f"BLOCKED DATA PATH: {path}")
            continue

        # Allow .env.example explicitly
        if path.name == ".env.example":
            continue

        if path.name in BLOCKED_BASENAMES or path.name.startswith(".env."):
            violations.append(f"BLOCKED SECRET FILE: {path}")
            continue

        if path.suffix.lower() in BLOCKED_EXTENSIONS:
            violations.append(f"BLOCKED EXTENSION ({path.suffix}): {path}")
            continue

        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            violations.append(f"FILE TOO LARGE ({size / 1e6:.2f} MB > 5 MB): {path}")
            continue

        # Scan text files for secret patterns
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for label, pattern in SECRET_PATTERNS:
            if pattern.search(content):
                violations.append(f"POTENTIAL SECRET ({label}) in {path}")

    if violations:
        print("\n[safety] COMMIT BLOCKED — security / public-repo violations found:")
        for v in violations:
            print(f"  - {v}")
        print("\nFix or unstage these files before committing.")
        return 1

    print(f"[safety] Passed ({len(files)} staged file(s) checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
