#!/usr/bin/env python3
"""Create a clearly labeled repeated-context Qasper replay input."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[15, 54])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 2 or len(set(args.indices)) != len(args.indices) or any(i < 0 for i in args.indices):
        parser.error("indices must be distinct and nonnegative; repetitions must be at least 2")
    if args.source.resolve() == args.output.resolve():
        parser.error("output must differ from the original dataset")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    source_bytes = args.source.read_bytes()
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if max(args.indices) >= len(rows):
        parser.error("sample index is outside the dataset")
    for index in args.indices:
        row = rows[index]
        if not isinstance(row.get("context"), str) or not row["context"]:
            raise ValueError(f"sample {index} has no text context")
        row["context"] = "\n\n".join([row["context"]] * args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "indices": args.indices,
        "context_repetitions": args.repetitions,
        "note": "Artificial repeated context for kernel replay, not a natural LongBench quality sample.",
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
