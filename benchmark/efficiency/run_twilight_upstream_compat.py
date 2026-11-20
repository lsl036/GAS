#!/usr/bin/env python3
"""Run an upstream efficiency script with a local, process-only FlashInfer JIT shim."""
import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('script', choices=('bench_gqa.py', 'bench_breakdown.py', 'bench_top_p.py', 'bench_gemv.py'))
    args, rest = parser.parse_known_args()

    import torch
    import flashinfer.jit as jit
    import flashinfer.utils as utils
    from flashinfer.jit import env
    from torch.utils.cpp_extension import load

    if not hasattr(utils, 'get_cuda_stream'):
        utils.get_cuda_stream = lambda device: torch.cuda.current_stream(device).cuda_stream
    if not hasattr(jit, 'load_cuda_ops'):
        from flashinfer.compilation_context import CompilationContext

        def load_cuda_ops(name, sources, extra_include_paths=None, **kwargs):
            if kwargs:
                raise TypeError(f'unsupported legacy JIT arguments: {sorted(kwargs)}')
            include = list(extra_include_paths or [])
            if name == 'ftka_ops':
                overlay = Path(__file__).resolve().parent / 'upstream_ftka_compat'
                sources = [overlay / 'src' / Path(source).name for source in sources]
                include = [overlay / 'include' if str(path).endswith('flash-topk-attention/csrc/include') else path
                           for path in include]
            # Sampling uses the CUDA toolkit CUB API; newer bundled CCCL
            # removes cub::Min/Max used by upstream sampling.cuh.
            include_keys = ('CUTLASS_INCLUDE_DIRS',) if name == 'twilight_sampling' else ('CUTLASS_INCLUDE_DIRS', 'CCCL_INCLUDE_DIRS')
            for key in include_keys:
                include.extend(getattr(env, key, []))
            flags = ['--expt-relaxed-constexpr', '-U__CUDA_NO_HALF_OPERATORS__',
                     '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                     '-U__CUDA_NO_HALF2_OPERATORS__', '-DFLASHINFER_HAS_UINT8_VEC_T']
            flags.extend(CompilationContext().get_nvcc_flags_list())
            return load(name=f'upstream_compat_{name}', sources=[str(x) for x in sources],
                        extra_include_paths=[str(x) for x in include], extra_cuda_cflags=flags,
                        verbose=False)

        jit.has_prebuilt_ops = False
        jit.load_cuda_ops = load_cuda_ops
        jit.FLASHINFER_INCLUDE_DIR = env.FLASHINFER_INCLUDE_DIR
        jit.FLASHINFER_CSRC_DIR = env.FLASHINFER_CSRC_DIR
    root = args.upstream.resolve()
    script = root / 'benchmark' / 'efficiency' / args.script
    sys.path[:0] = [str(script.parent), str(root), str(root / 'flash-topk-attention')]
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
