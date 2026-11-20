"""Offline regression checks for the paper reproduction package."""

import hashlib
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproduction"


def run(*args):
    return subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


class ReproductionPackageTest(unittest.TestCase):
    def test_snapshot_manifests(self):
        for directory in ("figure_inputs", "reference_results", "source_snapshot"):
            base = REPRO / directory
            manifest = json.loads((base / "manifest.json").read_text())
            self.assertTrue(manifest["files"])
            for entry in manifest["files"]:
                with self.subTest(directory=directory, path=entry["path"]):
                    path = base / entry["path"]
                    self.assertTrue(path.is_file())
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])

    def test_snapshot_and_audit(self):
        verify = run(REPRO / "figures.py", "verify")
        self.assertEqual(verify.returncode, 0, verify.stderr)
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "audit.md"
            result = run(REPRO / "audit.py", "--output", report)
            self.assertEqual(result.returncode, 0, result.stderr)
            text = report.read_text()
            self.assertIn("| Full attention | 64.58 | 60.58", text)
            self.assertIn("| DS KV-calibrated | 42.40 | 42.40 | 1458.00 | 1757.92", text)
            self.assertIn("## Fig. 6", text)
            self.assertIn("## Fig. 7", text)
            self.assertIn("## Fig. 8", text)

    def test_render_rejects_protected_and_occupied_destinations(self):
        protected = REPRO / "figure_inputs" / "new-output"
        result = run(REPRO / "figures.py", "render", "--figures", "fig1", "--output", protected)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not be inside", result.stderr)
        self.assertFalse(protected.exists())
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("do not change")
            result = run(REPRO / "figures.py", "render", "--figures", "fig1", "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(sentinel.read_text(), "do not change")

    def test_passkey_dry_run_resolves_ds_channel_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "passkey"
            result = run(
                REPRO / "run_benchmark.py", "--dry-run", "passkey", "--model", "/tmp/model",
                "--family", "llama31", "--selector", "ds", "--fixed-length", "8000", "--output", output,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(ROOT / "benchmark/configs/ds_channel_llama31_q.json"), result.stdout)
            self.assertIn("--skip-first-records 64", result.stdout)
            self.assertFalse(output.exists())

    def test_fig4a_and_qasper_dry_runs_do_not_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fig4 = run(
                REPRO / "run_benchmark.py", "--dry-run", "fig4a", "--model", "/tmp/model",
                "--channel-config", "/tmp/channels", "--calibration-data-root", "/tmp/cal",
                "--formal-data-root", "/tmp/formal", "--output", root / "fig4a",
            )
            self.assertEqual(fig4.returncode, 0, fig4.stderr)
            self.assertIn("run_fig4a.py", fig4.stdout)
            qasper = run(
                REPRO / "run_benchmark.py", "--dry-run", "qasper", "--context", "32k",
                "--batch", "64", "--timer", "author", "--output", root / "qasper",
            )
            self.assertEqual(qasper.returncode, 0, qasper.stderr)
            self.assertIn("--attention-only", qasper.stdout)
            self.assertFalse((root / "fig4a").exists())
            self.assertFalse((root / "qasper").exists())

    def test_source_snapshot_dry_run_uses_snapshot_config(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = REPRO / "source_snapshot"
            result = run(
                snapshot / "reproduction/run_benchmark.py", "--dry-run", "passkey", "--model", "/tmp/model",
                "--family", "llama31", "--selector", "ds", "--fixed-length", "8000", "--output", Path(temp) / "run",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(snapshot / "benchmark/configs/ds_channel_llama31_q.json"), result.stdout)

    def test_package_documentation_present(self):
        for name in ("README.md", "PROVENANCE.md", "PIPELINES.md"):
            self.assertTrue((REPRO / name).is_file())

    def test_development_package_name_and_discovery(self):
        result = run(ROOT / "setup.py", "--name")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "gas-attention-research")
        self.assertTrue((ROOT / "gas_attention/pyimpl/attention.py").is_file())
        self.assertFalse((ROOT / "twilight").exists())
        self.assertTrue((REPRO / "source_snapshot/twilight/pyimpl/attention.py").is_file())

    def test_active_python_imports_do_not_use_old_package(self):
        excluded = {"vendor", "dataset", "datasets", "logs", "__pycache__"}
        for base in (ROOT / "gas_attention", ROOT / "benchmark"):
            for path in base.rglob("*.py"):
                rel = path.relative_to(ROOT)
                if any(part in excluded or part.startswith(("results", "runs")) for part in rel.parts):
                    continue
                with self.subTest(path=str(rel)):
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            self.assertNotEqual(node.module.split(".")[0], "twilight", str(rel))
                        elif isinstance(node, ast.Import):
                            self.assertTrue(all(alias.name.split(".")[0] != "twilight" for alias in node.names), str(rel))


if __name__ == "__main__":
    unittest.main()
