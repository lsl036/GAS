import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmark"


def summarize_profile(profile_path: Path, summary_path: Path, skip_first_records: int = 0) -> dict:
    stage_values = defaultdict(list)
    max_selected = []
    avg_selected = []
    seq_lens = []
    total_values = []
    gqa_values = defaultdict(list)

    record_idx = 0
    with open(profile_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record_idx += 1
            if record_idx <= skip_first_records:
                continue
            record = json.loads(line)
            seq_lens.append(record["seq_len"])
            max_selected.append(record["max_selected"])
            avg_selected.append(record["avg_selected"])
            total_values.append(record["total_us"])
            for stage, value in record["stage_times_us"].items():
                stage_values[stage].append(value)
            for metric, value in record.get("gqa", {}).items():
                if isinstance(value, (int, float)):
                    gqa_values[metric].append(value)

    def stats(values):
        values = sorted(values)
        if not values:
            return {}
        p50_idx = int((len(values) - 1) * 0.50)
        p90_idx = int((len(values) - 1) * 0.90)
        return {
            "count": len(values),
            "mean_us": statistics.fmean(values),
            "p50_us": values[p50_idx],
            "p90_us": values[p90_idx],
            "max_us": values[-1],
        }

    total_mean = statistics.fmean(total_values) if total_values else 0.0
    stages = {}
    for stage, values in sorted(stage_values.items()):
        stage_stats = stats(values)
        stage_stats["mean_pct_total"] = (
            stage_stats["mean_us"] / total_mean * 100 if total_mean else 0.0
        )
        stages[stage] = stage_stats

    summary = {
        "records": len(total_values),
        "seq_len": stats(seq_lens),
        "max_selected": stats(max_selected),
        "avg_selected": stats(avg_selected),
        "total_decode_attention": stats(total_values),
        "stages": stages,
        "gqa": {metric: stats(values) for metric, values in sorted(gqa_values.items())},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def print_summary(summary: dict) -> None:
    print(f"profile records: {summary['records']}")
    print(
        "total_decode_attention mean: "
        f"{summary['total_decode_attention'].get('mean_us', 0.0):.3f} us"
    )
    print("stage breakdown:")
    for stage, stats in sorted(
        summary["stages"].items(),
        key=lambda item: item[1].get("mean_us", 0.0),
        reverse=True,
    ):
        print(
            f"  {stage}: mean={stats['mean_us']:.3f} us, "
            f"p50={stats['p50_us']:.3f} us, p90={stats['p90_us']:.3f} us, "
            f"max={stats['max_us']:.3f} us, pct={stats['mean_pct_total']:.2f}%"
        )
    if summary.get("gqa"):
        print("gqa metrics:")
        for metric, stats in sorted(summary["gqa"].items()):
            print(
                f"  {metric}: mean={stats['mean_us']:.3f}, "
                f"p50={stats['p50_us']:.3f}, p90={stats['p90_us']:.3f}, "
                f"max={stats['max_us']:.3f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data2/lsl_data/models/longchat-7b-v1.5-32k",
    )
    parser.add_argument(
        "--config",
        default=str(BENCHMARK_DIR / "configs/config_quest_twi_real_int4_estimator.json"),
    )
    parser.add_argument("--fixed-length", type=int, default=16000)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_DIR / "results_passkey_ours/profile_passkey_16k"),
    )
    parser.add_argument(
        "--profile-output",
        default=str(BENCHMARK_DIR / "results_passkey_ours/profile_passkey_16k/profile.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        default=str(BENCHMARK_DIR / "results_passkey_ours/profile_passkey_16k/profile_summary.json"),
    )
    parser.add_argument(
        "--torch-extensions-dir",
        default="/data/tmp/torch_ext_ours_int4_profile",
    )
    parser.add_argument(
        "--skip-first-records",
        type=int,
        default=64,
        help="Drop early attention records from summary to reduce JIT/warmup noise.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    profile_path = Path(args.profile_output)
    summary_path = Path(args.summary_output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TWILIGHT_PROFILE"] = "1"
    env["TWILIGHT_PROFILE_OUTPUT"] = str(profile_path)
    env["TWILIGHT_PROFILE_MODEL"] = args.model
    env["TWILIGHT_PROFILE_CONFIG"] = args.config
    env["TORCH_EXTENSIONS_DIR"] = args.torch_extensions_dir

    command = [
        sys.executable,
        str(BENCHMARK_DIR / "passkey/passkey.py"),
        "-m",
        args.model,
        "--iterations",
        str(args.iterations),
        "--fixed-length",
        str(args.fixed_length),
        "--algo-config-path",
        args.config,
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=str(BENCHMARK_DIR), env=env, check=True)

    summary = summarize_profile(profile_path, summary_path, args.skip_first_records)
    print_summary(summary)
    print(f"profile jsonl: {profile_path}")
    print(f"profile summary: {summary_path}")


if __name__ == "__main__":
    main()
