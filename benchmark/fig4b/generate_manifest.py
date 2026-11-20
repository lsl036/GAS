#!/usr/bin/env python3
"""Create the shared 50-case Passkey manifest for Fig. 4(b)."""

import argparse
import json
import random
from pathlib import Path


DEPTHS = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--repeats-per-depth", type=int, default=5)
    args = parser.parse_args()
    if args.repeats_per_depth <= 0:
        raise ValueError("--repeats-per-depth must be positive")

    rng = random.Random(args.seed)
    cases = []
    for depth in DEPTHS:
        for repeat in range(args.repeats_per_depth):
            cases.append(
                {
                    "case_id": f"d{depth:02d}_r{repeat:02d}",
                    "depth_percent": depth,
                    "repeat": repeat,
                    "pass_key": rng.randint(1, 50000),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "depths": list(DEPTHS),
                "repeats_per_depth": args.repeats_per_depth,
                "n_cases": len(cases),
                "cases": cases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
