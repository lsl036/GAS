import atexit
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch


class StageTimer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.stage_times_us: Dict[str, float] = {}
        self._cuda_events = {}
        self._wall_starts = {}

    def cuda_start(self, name: str) -> None:
        if not self.enabled:
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._cuda_events[name] = (start, end)

    def cuda_end(self, name: str) -> None:
        if not self.enabled:
            return
        start, end = self._cuda_events[name]
        end.record()

    def finalize(self) -> None:
        if not self.enabled:
            return
        for name, (start, end) in self._cuda_events.items():
            end.synchronize()
            self.stage_times_us[name] = (
                self.stage_times_us.get(name, 0.0) + start.elapsed_time(end) * 1000
            )
        self._cuda_events.clear()

    def wall_start(self, name: str) -> None:
        if self.enabled:
            self._wall_starts[name] = time.perf_counter()

    def wall_end(self, name: str) -> None:
        if not self.enabled:
            return
        elapsed_us = (time.perf_counter() - self._wall_starts[name]) * 1e6
        self.stage_times_us[name] = self.stage_times_us.get(name, 0.0) + elapsed_us


class TwilightProfiler:
    def __init__(self):
        self.enabled = os.environ.get("TWILIGHT_PROFILE", "0") == "1"
        self.output_path = os.environ.get("TWILIGHT_PROFILE_OUTPUT")
        self.write_raw = (
            bool(self.output_path)
            and os.environ.get("TWILIGHT_PROFILE_RAW", "1") == "1"
        )
        self.model_name = os.environ.get("TWILIGHT_PROFILE_MODEL", "")
        self.config_name = os.environ.get("TWILIGHT_PROFILE_CONFIG", "")
        self._records = []
        self._query_records = None

        if self.enabled and self.write_raw:
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            atexit.register(self.flush)

    def new_timer(self) -> StageTimer:
        return StageTimer(self.enabled)

    def record(self, record: Dict) -> None:
        if not self.enabled:
            return
        record["model_name"] = self.model_name
        record["config_name"] = self.config_name
        if self.write_raw:
            self._records.append(record)
        if self._query_records is not None:
            self._query_records.append(dict(record))
        if self.write_raw and len(self._records) >= 128:
            self.flush()

    def begin_query(self) -> None:
        if self.enabled:
            self._query_records = []

    def end_query(self):
        if not self.enabled:
            return []
        records = self._query_records or []
        self._query_records = None
        return records

    def flush(self) -> None:
        if not self.enabled or not self.write_raw or not self.output_path or not self._records:
            return
        with open(self.output_path, "a", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        self._records.clear()


profiler = TwilightProfiler()
