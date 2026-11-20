#!/usr/bin/env python3
"""Recover the measured vector16 source before the later host alignment guard."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "csrc/src/gas_warp_decode.cu"
result = ROOT / "benchmark/results_efficiency/gas_vector16_full_v1"
expected = json.loads((result / "config.json").read_text())["source_sha256"]
text = source.read_text()
header = "#include <cstdint>\n"
guard = '''  if (mode >= 8) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "vector inputs must be 16-byte aligned");
  }
'''
if text.count(header) != 1 or text.count(guard) != 1:
    raise ValueError("source no longer matches the known host-only change")
snapshot = text.replace(header, "", 1).replace(guard, "", 1).encode()
actual = hashlib.sha256(snapshot).hexdigest()
if actual != expected:
    raise ValueError(f"snapshot digest mismatch: {actual} != {expected}")
target = result / "source_snapshot.cu"
if target.exists() and target.read_bytes() != snapshot:
    raise ValueError(f"refusing to overwrite {target}")
target.write_bytes(snapshot)
print(f"verified {actual}: {target}")
