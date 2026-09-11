from __future__ import annotations
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from .config import Config, load_config
from .recognition_model import (
    TrainedModel,
    model_param_count,
    model_size_on_disk_bytes,
    predict,
)

@dataclass
class EfficiencyReport:
    model_name: str
    param_count: int
    size_on_disk_bytes: int
    size_on_disk_mb: float
    peak_memory_bytes: int
    peak_memory_mb: float
    mean_latency_ms: float
    std_latency_ms: float
    n_repeats: int
    target_device_label: str

def measure_single_query_latency(
    model: TrainedModel, X_single: pd.DataFrame, n_repeats: int = 50
) -> tuple[float, float]:
    _ = predict(model, X_single)

    timings_ms: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = predict(model, X_single)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(timings_ms)), float(np.std(timings_ms))

def measure_peak_memory_bytes(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)

def measure_model_efficiency(
    model: TrainedModel,
    X_sample_single: pd.DataFrame,
    X_sample_batch: pd.DataFrame,
    cfg: Optional[Config] = None,
) -> EfficiencyReport:
    cfg = cfg or load_config()
    eff_cfg = cfg.raw["efficiency"]
    n_repeats = int(eff_cfg["n_latency_repeats"])

    size_bytes = model_size_on_disk_bytes(model)
    param_count = model_param_count(model)
    mean_lat_ms, std_lat_ms = measure_single_query_latency(model, X_sample_single, n_repeats)
    peak_mem = measure_peak_memory_bytes(lambda: predict(model, X_sample_batch))

    return EfficiencyReport(
        model_name=model.name,
        param_count=param_count,
        size_on_disk_bytes=size_bytes,
        size_on_disk_mb=size_bytes / (1024 * 1024),
        peak_memory_bytes=peak_mem,
        peak_memory_mb=peak_mem / (1024 * 1024),
        mean_latency_ms=mean_lat_ms,
        std_latency_ms=std_lat_ms,
        n_repeats=n_repeats,
        target_device_label=str(eff_cfg["target_device_label"]),
    )

def report_to_dict(report: EfficiencyReport) -> Dict[str, object]:
    return {
        "model_name": report.model_name,
        "param_count": report.param_count,
        "size_on_disk_mb": round(report.size_on_disk_mb, 4),
        "peak_memory_mb": round(report.peak_memory_mb, 4),
        "mean_latency_ms": round(report.mean_latency_ms, 4),
        "std_latency_ms": round(report.std_latency_ms, 4),
        "n_repeats": report.n_repeats,
        "target_device_label": report.target_device_label,
    }