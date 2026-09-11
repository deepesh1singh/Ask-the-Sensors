from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

RAW_STREAM_COLUMNS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TARGET_HZ = 25.0
DEFAULT_WINDOW_SECONDS = 60.0

@dataclass
class RawStreamValidationError(ValueError):
    message: str
    def __str__(self) -> str:  
        return self.message

def load_raw_stream(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise RawStreamValidationError(f"Raw stream file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc: 
        raise RawStreamValidationError(f"Could not parse {path.name} as CSV: {exc}") from exc

    missing = [c for c in RAW_STREAM_COLUMNS if c not in df.columns]
    if missing:
        raise RawStreamValidationError(
            f"{path.name} is missing required column(s) {missing}. "
            f"Expected columns: {RAW_STREAM_COLUMNS}."
        )
    for col in RAW_STREAM_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) < n_before:
        pass
    if len(df) < 2:
        raise RawStreamValidationError(
            f"{path.name} has fewer than 2 valid timestamped rows after parsing; "
            "cannot resample or window a recording this short."
        )
    return df

def resample_to_25hz(
    raw: pd.DataFrame,
    target_hz: float = TARGET_HZ,
    max_gap_seconds: float = 5.0,
) -> pd.DataFrame:
    if target_hz <= 0:
        raise ValueError(f"target_hz must be positive, got {target_hz}")
    t0, t1 = float(raw["timestamp"].iloc[0]), float(raw["timestamp"].iloc[-1])
    if t1 <= t0:
        raise RawStreamValidationError(
            "Raw stream timestamps are non-increasing after sorting; cannot resample."
        )
    step = 1.0 / target_hz
    n_out = int(np.floor((t1 - t0) / step)) + 1
    grid = t0 + step * np.arange(n_out)

    out = pd.DataFrame({"timestamp": grid})
    src_t = raw["timestamp"].to_numpy(dtype=float)
    for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        src_v = raw[col].to_numpy(dtype=float)
        valid = ~np.isnan(src_v)
        if valid.sum() < 2:
            out[col] = np.nan
            continue
        interp = np.interp(grid, src_t[valid], src_v[valid], left=np.nan, right=np.nan)
        valid_t = src_t[valid]
        gap_before_idx = np.searchsorted(valid_t, grid, side="right") - 1
        gap_after_idx = np.clip(gap_before_idx + 1, 0, len(valid_t) - 1)
        gap_before_idx = np.clip(gap_before_idx, 0, len(valid_t) - 1)
        local_gap = valid_t[gap_after_idx] - valid_t[gap_before_idx]
        in_big_gap = local_gap > max_gap_seconds
        interp = np.where(in_big_gap, np.nan, interp)
        out[col] = interp
    return out

def _magnitude_stats(mag: np.ndarray) -> Dict[str, float]:
    mag = mag[~np.isnan(mag)]
    keys = ["mean", "std", "moment3", "moment4", "percentile25", "percentile50",
            "percentile75", "value_entropy", "time_entropy"]
    if len(mag) < 2:
        return {k: np.nan for k in keys}
    mean = float(np.mean(mag))
    std = float(np.std(mag))
    if std > 1e-9:
        z = (mag - mean) / std
        moment3 = float(np.mean(z ** 3))
        moment4 = float(np.mean(z ** 4))
    else:
        moment3, moment4 = 0.0, 0.0
    p25, p50, p75 = (float(v) for v in np.percentile(mag, [25, 50, 75]))
    n_bins = int(np.clip(len(mag) // 4, 2, 16))
    hist, _ = np.histogram(mag, bins=n_bins)
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    value_entropy = float(-np.sum(p * np.log2(p))) if len(p) else 0.0
    shifted = mag - mag.min() if mag.min() < 0 else mag
    s = shifted.sum()
    if s > 1e-9:
        p_t = shifted / s
        p_t = p_t[p_t > 0]
        time_entropy = float(-np.sum(p_t * np.log2(p_t)))
    else:
        time_entropy = 0.0
    return {
        "mean": mean, "std": std, "moment3": moment3, "moment4": moment4,
        "percentile25": p25, "percentile50": p50, "percentile75": p75,
        "value_entropy": value_entropy, "time_entropy": time_entropy,
    }

def _spectral_features(mag: np.ndarray, fs: float) -> Dict[str, float]:
    mag = mag[~np.isnan(mag)]
    n = len(mag)
    keys = [f"log_energy_band{i}" for i in range(5)] + ["spectral_entropy"]
    if n < 8:
        return {k: np.nan for k in keys}
    windowed = (mag - mag.mean()) * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    nyquist = fs / 2.0
    edges = np.linspace(0, nyquist, 6)
    band_energy = []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi) if i < 4 else (freqs >= lo) & (freqs <= hi)
        e = spectrum[mask].sum()
        band_energy.append(float(np.log(e + 1e-9)))
    total = spectrum.sum()
    if total > 1e-9:
        p = spectrum / total
        p = p[p > 0]
        spectral_entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(p))) if len(p) > 1 else 0.0
    else:
        spectral_entropy = 0.0
    out = {f"log_energy_band{i}": v for i, v in enumerate(band_energy)}
    out["spectral_entropy"] = spectral_entropy
    return out

def _autocorrelation_features(mag: np.ndarray, fs: float) -> Dict[str, float]:
    mag = mag[~np.isnan(mag)]
    n = len(mag)
    if n < 8:
        return {"period": np.nan, "normalized_ac": np.nan}
    x = mag - mag.mean()
    var = float(np.dot(x, x))
    if var < 1e-9:
        return {"period": 0.0, "normalized_ac": 0.0}
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / var
    min_lag = max(1, int(0.2 * fs))
    if min_lag >= len(ac) - 1:
        return {"period": 0.0, "normalized_ac": 0.0}
    search = ac[min_lag:]
    peak_idx = int(np.argmax(search)) + min_lag
    return {"period": float(peak_idx / fs), "normalized_ac": float(ac[peak_idx])}

def _axis_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    keys = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "ro_xy", "ro_xz", "ro_yz"]
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 2:
        return {k: np.nan for k in keys}
    out = {
        "mean_x": float(np.mean(x)), "mean_y": float(np.mean(y)), "mean_z": float(np.mean(z)),
        "std_x": float(np.std(x)), "std_y": float(np.std(y)), "std_z": float(np.std(z)),
    }
    for name, a, b in [("ro_xy", x, y), ("ro_xz", x, z), ("ro_yz", y, z)]:
        sa, sb = np.std(a), np.std(b)
        if sa > 1e-9 and sb > 1e-9:
            out[name] = float(np.corrcoef(a, b)[0, 1])
        else:
            out[name] = 0.0
    return out

def extract_window_features(window: pd.DataFrame, fs: float = TARGET_HZ) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for prefix, cols in [("raw_acc", ("acc_x", "acc_y", "acc_z")), ("proc_gyro", ("gyro_x", "gyro_y", "gyro_z"))]:
        x = window[cols[0]].to_numpy(dtype=float)
        y = window[cols[1]].to_numpy(dtype=float)
        z = window[cols[2]].to_numpy(dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        mag = np.full(len(x), np.nan)
        mag[valid] = np.sqrt(x[valid] ** 2 + y[valid] ** 2 + z[valid] ** 2)

        for k, v in _magnitude_stats(mag).items():
            out[f"{prefix}:magnitude_stats:{k}"] = v
        for k, v in _spectral_features(mag, fs).items():
            out[f"{prefix}:magnitude_spectrum:{k}"] = v
        for k, v in _autocorrelation_features(mag, fs).items():
            out[f"{prefix}:magnitude_autocorrelation:{k}"] = v
        for k, v in _axis_features(x, y, z).items():
            out[f"{prefix}:3d:{k}"] = v
    return out

def raw_stream_to_feature_frame(
    raw: pd.DataFrame,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    target_hz: float = TARGET_HZ,
    min_valid_fraction: float = 0.5,
) -> pd.DataFrame:
    resampled = resample_to_25hz(raw, target_hz=target_hz)
    samples_per_window = int(round(window_seconds * target_hz))
    if samples_per_window < 8:
        raise ValueError(
            f"window_seconds={window_seconds} at target_hz={target_hz} gives only "
            f"{samples_per_window} samples/window; need at least 8 for spectral features."
        )
    rows: List[Dict[str, float]] = []
    n = len(resampled)
    for start_idx in range(0, n, samples_per_window):
        end_idx = min(start_idx + samples_per_window, n)
        chunk = resampled.iloc[start_idx:end_idx]
        if len(chunk) < samples_per_window // 2:
            continue
        valid_fraction = float(chunk["acc_x"].notna().mean())
        if valid_fraction < min_valid_fraction:
            continue
        feats = extract_window_features(chunk, fs=target_hz)
        feats["timestamp"] = float(chunk["timestamp"].iloc[0])
        rows.append(feats)
    if not rows:
        raise RawStreamValidationError(
            "No usable windows survived resampling and gap-filtering -- the "
            "recording may be too short, too sparse, or entirely inside one "
            "large gap."
        )
    frame = pd.DataFrame(rows)
    cols = ["timestamp"] + [c for c in frame.columns if c != "timestamp"]
    return frame[cols].reset_index(drop=True)

def build_userdata_from_raw_stream(
    path: Path,
    uuid: str,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    target_hz: float = TARGET_HZ,
):
    from .data_io import UserData
    raw = load_raw_stream(path)
    frame = raw_stream_to_feature_frame(raw, window_seconds=window_seconds, target_hz=target_hz)
    timestamps = frame["timestamp"].to_numpy(dtype="int64")
    features = frame.drop(columns=["timestamp"])
    features.index = timestamps
    labels = pd.DataFrame(index=timestamps)
    label_source = np.full(len(timestamps), -1, dtype="int8")

    return UserData(
        uuid=uuid,
        timestamps=timestamps,
        features=features,
        labels=labels,
        label_source=label_source,
    )