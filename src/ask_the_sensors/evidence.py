from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from .aggregation import Interval
from .config import Config, load_config


def _get(summary: Dict[str, float], *candidates: str) -> Optional[float]:
    for c in candidates:
        if c in summary and summary[c] == summary[c]:  # not NaN
            return summary[c]
    return None

def describe_signal(interval: Interval) -> str:
    """Deterministically describe the dominant signal characteristics of an
    interval's averaged accel/gyro feature summary, in plain language."""
    s = interval.signal_summary
    parts: List[str] = []

    acc_mean = _get(s, "raw_acc:magnitude_stats:mean", "raw_acc:3d:mean")
    acc_std = _get(s, "raw_acc:magnitude_stats:std", "raw_acc:3d:std")
    gyro_mean = _get(s, "proc_gyro:magnitude_stats:mean", "proc_gyro:3d:mean")
    gyro_std = _get(s, "proc_gyro:magnitude_stats:std", "proc_gyro:3d:std")
    period = _get(
        s,
        "raw_acc:autocorrelation:period",
        "raw_acc:autocorrelation:normalized_ac",
    )

    if acc_std is not None:
        if acc_std < 0.05:
            parts.append(f"very low accelerometer variance (std={acc_std:.3f})")
        elif acc_std < 0.3:
            parts.append(f"low-to-moderate accelerometer variance (std={acc_std:.3f})")
        elif acc_std < 1.0:
            parts.append(f"moderate accelerometer variance (std={acc_std:.3f})")
        else:
            parts.append(f"high accelerometer variance (std={acc_std:.3f})")

    if acc_mean is not None:
        parts.append(f"mean accelerometer magnitude {acc_mean:.3f}")

    if period is not None and period > 0:
        approx_hz = 1.0 / period if period > 1e-6 else None
        if approx_hz is not None and 0.3 <= approx_hz <= 5.0:
            parts.append(f"periodic component at ~{approx_hz:.1f} Hz (cadence-like)")

    if gyro_std is not None:
        if gyro_std < 0.1:
            parts.append(f"low gyroscope variance (std={gyro_std:.3f})")
        elif gyro_std < 0.5:
            parts.append(f"moderate gyroscope variance (std={gyro_std:.3f})")
        else:
            parts.append(f"high gyroscope variance (std={gyro_std:.3f})")

    if gyro_mean is not None:
        parts.append(f"mean gyroscope magnitude {gyro_mean:.3f}")

    if not parts:
        parts.append("signal summary unavailable for this interval (feature columns missing)")

    return "; ".join(parts)

@dataclass
class EvidenceBlock:
    timestamps: str
    sensor_modality: str
    sensor_channels: str
    def to_dict(self) -> Dict[str, str]:
        return {
            "Timestamp(s)": self.timestamps,
            "Sensor Modality": self.sensor_modality,
            "Sensor Channel(s)": self.sensor_channels,
        }

def format_timestamp_range(start_s: int, end_s: int) -> str:
    return f"{start_s} to {end_s} (seconds from start)"

def format_timestamp_ranges(intervals: List[Interval]) -> str:
    if not intervals:
        return "N/A"
    return ", ".join(format_timestamp_range(iv.start_s, iv.end_s) for iv in intervals)

def build_evidence(
    intervals: List[Interval],
    modality: str = "both",
    channels: str = "All",
) -> EvidenceBlock:
    modality_str = {
        "accel": "Accelerometer",
        "gyro": "Gyroscope",
        "both": "Accelerometer, Gyroscope",
    }.get(modality, modality)

    return EvidenceBlock(
        timestamps=format_timestamp_ranges(intervals),
        sensor_modality=modality_str if intervals else "N/A",
        sensor_channels=channels if intervals else "N/A",
    )

def build_na_evidence() -> EvidenceBlock:
    return EvidenceBlock(timestamps="N/A", sensor_modality="N/A", sensor_channels="N/A")