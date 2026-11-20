#!/usr/bin/env python3
"""Create and verify a compact, immutable research-source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent / "source_snapshot"
TREES = (
    "gas_attention", "csrc", "flash-topk-attention", "benchmark/configs",
    "benchmark/efficiency", "benchmark/LongBench", "benchmark/PG19",
    "benchmark/passkey", "benchmark/RULER/scripts", "benchmark/RULER/fig4a",
    "benchmark/fig4b", "benchmark/scripts",
)
EXCLUDE = {".git", "__pycache__", ".pytest_cache", "vendor", "dataset", "datasets", "logs", "build", "dist", "results"}
SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".sh", ".json", ".yaml", ".yml", ".txt", ".md", ".cmake", ".toml"}
EXTRA = ("setup.py", "requirements.txt", "README.md", ".gitmodules", "benchmark/RULER/run_selector_matrix.sh", "reproduction/run_benchmark.py")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_paths():
    files = {Path(name) for name in EXTRA if (ROOT / name).is_file()}
    for tree in TREES:
        base = ROOT / tree
        if not base.is_dir():
            raise FileNotFoundError(base)
        for current, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in EXCLUDE and not d.startswith("results_")]
            if Path(current).relative_to(ROOT) == Path("benchmark/RULER/scripts/data/synthetic"):
                dirs[:] = [d for d in dirs if d != "json"]
            for name in names:
                path = Path(current) / name
                if path.is_file() and not path.is_symlink() and (path.suffix.lower() in SUFFIXES or path.name in {"LICENSE", "CMakeLists.txt"}):
                    files.add(path.relative_to(ROOT))
    return sorted(files)


def create():
    if DEST.exists():
        raise FileExistsError(f"refusing to overwrite {DEST}")
    files = selected_paths()
    if not files:
        raise ValueError("no source files selected")
    DEST.mkdir()
    entries = []
    for rel in files:
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
        entries.append({"path": str(rel), "sha256": checksum(target), "bytes": target.stat().st_size})
    (DEST / "manifest.json").write_text(json.dumps({"files": entries}, indent=2) + "\n")
    print(f"snapshotted {len(entries)} source files to {DEST}")


def verify():
    entries = json.loads((DEST / "manifest.json").read_text())["files"]
    failures = [e["path"] for e in entries if not (DEST / e["path"]).is_file() or checksum(DEST / e["path"]) != e["sha256"]]
    if failures:
        raise ValueError(f"source snapshot missing or changed: {failures}")
    print(f"verified {len(entries)} source files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    {"create": create, "verify": verify}[args.action]()
