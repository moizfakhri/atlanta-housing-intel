"""Tests for public-repo safety scanner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_safety.py"
spec = importlib.util.spec_from_file_location("check_repo_safety", SCRIPT_PATH)
assert spec and spec.loader
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def test_secret_patterns_catch_common_keys() -> None:
    # Build synthetic tokens dynamically so GitHub Push Protection & local scanner stay clean
    samples = [
        ("AWS Access Key", "AKIA" + "IOSFODNN7EXAMPLE"),
        ("GitHub Token", "ghp_" + "1234567890abcdefghijklmnopQRSTUV"),
        ("Google / Gemini API Key", "AIza" + "SyD-1234567890abcdefghijklmnopqrstuvw"),
        ("Hardcoded API Secret Assignment", "api_key = " + '"super_secret_token_value_12345"'),
    ]
    pattern_map = dict(safety.SECRET_PATTERNS)
    for label, text in samples:
        assert pattern_map[label].search(text), f"Expected pattern {label} to match"


def test_blocked_extensions_cover_data_and_secrets() -> None:
    for ext in [".env", ".parquet", ".duckdb", ".pmtiles", ".csv", ".shp", ".pt"]:
        assert ext in safety.BLOCKED_EXTENSIONS
