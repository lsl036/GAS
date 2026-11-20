#!/usr/bin/env python3
"""Record unmodified upstream Quest/Quest-Twi breakdown component timings."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT.parent / 'Twilight_Upstream_20261113'
WRAPPER = ROOT / 'benchmark/efficiency/run_twilight_upstream_compat.py'


def parse(stdout):
    result = {}
    current = None
    for line in stdout.splitlines():
        match = re.fullmatch(r'batch_size: (\d+)', line.strip())
        if match:
            current = int(match.group(1))
            result[current] = {}
        match = re.fullmatch(r'\[(token selector|spgemv|top p|attention)\] time: ([0-9.]+) us', line.strip())
        if match and current is not None:
            result[current][match.group(1)] = float(match.group(2))
    if set(result) != {32, 64, 96, 128} or any(len(v) != 4 for v in result.values()):
        raise ValueError(f'incomplete upstream results: {result}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=10)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--seq-len', type=int, default=32764)
    p.add_argument('--b0', type=int, default=8192)
    p.add_argument('--b1', type=int, default=721)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    records = {}
    for mode in ('quest', 'quest-twi'):
        command = [sys.executable, '-u', str(WRAPPER), '--upstream', str(UPSTREAM),
                   'bench_breakdown.py', '--batch-sizes', '32,64,96,128', '--seq-len', str(a.seq_len),
                   '--b0', str(a.b0), '--b1', str(a.b1), '--mode', mode, '--attention-source', 'profile',
                   '--warmups', str(a.warmups), '--repeats', str(a.repeats)]
        print('RUN', mode, flush=True)
        proc = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout, flush=True)
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command)
        records[mode] = {'command': command, 'stdout': proc.stdout, 'components_us': parse(proc.stdout)}
    source = UPSTREAM / 'benchmark/efficiency/bench_breakdown.py'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'upstream_script': str(source),
                                    'upstream_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                    'compat_wrapper_sha256': hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
                                    'notes': ['upstream synthetic random components, not the Qasper trace',
                                              'profile attention uses a synthetic per-head distribution, not GQA union',
                                              'means as reported by the upstream timing helper'],
                                    'results': records}, indent=2) + '\n')
    print('WROTE', a.output, flush=True)


if __name__ == '__main__':
    main()
