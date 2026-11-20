import os
import subprocess
from pathlib import Path


GUARD = Path(__file__).with_name("fig6_gpu_guard.sh")


def run_case(tmp_path, gpu_script, command):
    (tmp_path / "logs").mkdir()
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\n" + gpu_script + "\n")
    smi.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", OUT=str(tmp_path), LABEL="test", CUDA_VISIBLE_DEVICES="0")
    return subprocess.run(["bash", "-c", f'source "$1"; {command}', "bash", str(GUARD)],
                          env=env, capture_output=True, text=True, timeout=10)


def test_guard_refuses_busy_gpu(tmp_path):
    result = run_case(tmp_path, "echo 999999", 'run_guarded phase touch "$OUT/started"')
    assert result.returncode != 0
    assert not (tmp_path / "started").exists()


def test_guard_runs_when_idle(tmp_path):
    result = run_case(tmp_path, ":", 'run_guarded phase bash -c "exit 0"')
    assert result.returncode == 0


def test_guard_stops_only_own_process_on_contention(tmp_path):
    result = run_case(tmp_path, 'if [[ -f "$OUT/contend" ]]; then echo 999999; fi',
                      'run_guarded phase bash -c \'touch "$OUT/contend"; exec sleep 20\'')
    assert result.returncode != 0
    assert "stopping only benchmark" in result.stderr
    assert "other_pids=999999" in (tmp_path / "logs/gpu_contention.log").read_text()


def test_guard_stops_own_process_if_monitor_fails(tmp_path):
    result = run_case(tmp_path, 'if [[ -f "$OUT/contend" ]]; then exit 1; fi',
                      'run_guarded phase bash -c \'touch "$OUT/contend"; exec sleep 20\'')
    assert result.returncode != 0
    assert "GPU monitor failed" in result.stderr
