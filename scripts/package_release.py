#!/usr/bin/env python3
"""Create a clean GitHub-ready ZIP and repository SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "dist", "build"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if relative.parts[:2] == ("results", "runs") and path.name != ".gitkeep":
        return False
    if relative.parts[:2] == ("results", "scale_validation_smoke"):
        return False
    return path.name != "MANIFEST_SHA256.txt" and path.suffix not in {".pyc", ".pyo"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "lp-lasso-4vfb-experiments-github.zip",
    )
    args = parser.parse_args()
    files = [path for path in sorted(ROOT.rglob("*")) if path.is_file() and included(path)]
    manifest_lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    manifest = ROOT / "MANIFEST_SHA256.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    files.append(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = ROOT.name
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(prefix) / path.relative_to(ROOT))
    print(args.output)


if __name__ == "__main__":
    main()
