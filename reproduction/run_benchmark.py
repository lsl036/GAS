#!/usr/bin/env python3
"""Re-run documented figure experiments into a new output directory.

Use --dry-run first. These are expensive research benchmarks, not plotting.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', default=sys.executable, help='Python interpreter with GAS and GPU dependencies')
    p.add_argument('--gpu', default='0')
    p.add_argument('--dry-run', action='store_true')
    sub = p.add_subparsers(dest='benchmark', required=True)

    s = sub.add_parser('passkey', help='One Fig.5 fixed-8192 Passkey/profiler cell')
    s.add_argument('--model', type=Path, required=True)
    s.add_argument('--family', choices=('llama31', 'qwen3'), required=True)
    s.add_argument('--selector', choices=('quest', 'ds'), required=True)
    s.add_argument('--fixed-length', type=int, choices=(8000, 16000), required=True)
    s.add_argument('--iterations', type=int, default=5)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--channel-config', type=Path)

    s = sub.add_parser('fig4a', help='Full LLaMA Fig.4(a) calibration and formal RULER pipeline')
    s.add_argument('--model', type=Path, required=True)
    s.add_argument('--model-name', default='llama3-8b-chat-128k')
    s.add_argument('--channel-config', type=Path, required=True)
    s.add_argument('--calibration-data-root', type=Path, required=True)
    s.add_argument('--formal-data-root', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)

    s = sub.add_parser('qasper', help='One Fig.6 paired GAS-FI/custom attention cell')
    s.add_argument('--context', choices=('16k', '32k'), required=True)
    s.add_argument('--batch', type=int, choices=(32, 64, 96, 128), required=True)
    s.add_argument('--timer', choices=('author', 'graph'), default='author')
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--trace-root', type=Path, default=ROOT / 'benchmark/results_efficiency/fig6_traces')
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to reuse nonempty output directory: {output}')
    common_env = os.environ.copy()
    common_env['CUDA_VISIBLE_DEVICES'] = args.gpu
    common_env['PYTHONPATH'] = str(ROOT) + os.pathsep + common_env.get('PYTHONPATH', '')
    common_env.setdefault('TORCH_EXTENSIONS_DIR', str(Path('/data/tmp/gas_reproduction_torch_ext')))
    command = [args.python]
    required = []
    config = None
    if args.benchmark == 'passkey':
        name = ('config_quest_twi_gqa_group_max_p080_exception16.json' if args.selector == 'quest' else
                f'config_ds_twi_gqa_group_max_p080_exception16_{args.family}.json')
        source = ROOT / 'benchmark/configs' / name
        config = json.loads(source.read_text())
        required.append(args.model)
        if args.selector == 'ds':
            channel = (args.channel_config or ROOT / 'benchmark/configs' / f'ds_channel_{args.family}_q.json').resolve()
            config['selector']['config_path'] = str(channel)
            required.append(channel)
        config_path = output / 'resolved_config.json'
        command += [str(ROOT / 'benchmark/efficiency/profile_passkey_sparse_attention.py'),
                    '--model', str(args.model.resolve()), '--fixed-length', str(args.fixed_length),
                    '--iterations', str(args.iterations), '--config', str(config_path),
                    '--output-dir', str(output / 'results'), '--profile-output', str(output / 'profile.jsonl'),
                    '--summary-output', str(output / 'profile_summary.json'),
                    '--torch-extensions-dir', common_env['TORCH_EXTENSIONS_DIR'], '--skip-first-records', '64']
    elif args.benchmark == 'fig4a':
        required.extend([args.model, args.channel_config, args.calibration_data_root, args.formal_data_root])
        command += [str(ROOT / 'benchmark/RULER/fig4a/run_fig4a.py'), '--model-path', str(args.model.resolve()),
                    '--model-name', args.model_name, '--selector', 'quest', '--channel-config', str(args.channel_config.resolve()),
                    '--calibration-data-root', str(args.calibration_data_root.resolve()),
                    '--formal-data-root', str(args.formal_data_root.resolve()), '--output-dir', str(output)]
    else:
        prefix = 'qasper16k' if args.context == '16k' else 'qasper32k_repeated'
        dirs = [args.trace_root / f'{prefix}_rate08_s{x}' for x in (15, 54)]
        required.extend(dirs)
        # The replay script fixes the trace location; do not imply --trace-root can change it.
        expected = ROOT / 'benchmark/results_efficiency/fig6_traces'
        if args.trace_root.resolve() != expected.resolve():
            p.error(f'current replay script requires trace-root {expected}')
        command += [str(ROOT / 'benchmark/efficiency/bench_qasper_gas_paired.py'), '--context', args.context,
                    '--batch', str(args.batch), '--timer', args.timer, '--attention-only',
                    '--warmups', '5', '--repeats', '50', '--rounds', '3', '--output-dir', str(output)]
    print('cwd:', ROOT)
    print('CUDA_VISIBLE_DEVICES:', args.gpu)
    print('PYTHONPATH:', common_env['PYTHONPATH'])
    print('command:', shlex.join(command))
    if config is not None:
        print('resolved config:', json.dumps(config, ensure_ascii=False))
    if args.dry_run:
        return
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'missing inputs: {missing}')
    if config is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'resolved_config.json').write_text(json.dumps(config, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=common_env, check=True)


if __name__ == '__main__':
    main()
