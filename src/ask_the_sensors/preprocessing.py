from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .config import Config, load_config
from .data_io import UserData, get_primary_activity_label, select_modality_features

@dataclass
class PreparedUser:
    uuid: str
    t_start: int                    
    rel_time_s: np.ndarray        
    X: pd.DataFrame                
    y: pd.Series                  
    gap_before: np.ndarray          

def compute_feature_nan_fractions(users: Dict[str, UserData], cfg: Config) -> pd.Series:
    prefixes = tuple(cfg.feature_prefixes.values())
    all_cols: Optional[pd.Index] = None
    nan_counts: Optional[pd.Series] = None
    total_counts = 0

    for ud in users.values():
        feats = select_modality_features(ud.features, cfg)
        if all_cols is None:
            all_cols = feats.columns
            nan_counts = pd.Series(0, index=all_cols, dtype=np.int64)
        else:
            feats = feats.reindex(columns=all_cols)
        nan_counts = nan_counts.add(feats.isna().sum(axis=0), fill_value=0)
        total_counts += len(feats)

    if nan_counts is None or total_counts == 0:
        return pd.Series(dtype=np.float64)
    return nan_counts / float(total_counts)

def resolve_available_activity_classes(
    users: Dict[str, UserData], cfg: Config, min_examples: int = 1
) -> Tuple[List[str], List[str]]:
    counts = {c: 0 for c in cfg.activity_classes}
    for ud in users.values():
        present = [c for c in cfg.activity_classes if c in ud.labels.columns]
        if not present:
            continue
        positive_counts = (ud.labels[present] == 1.0).sum(axis=0)
        for c in present:
            counts[c] += int(positive_counts[c])

    available = [c for c in cfg.activity_classes if counts[c] >= min_examples]
    unavailable = [c for c in cfg.activity_classes if counts[c] < min_examples]
    return available, unavailable

def select_stable_feature_columns(
    users: Dict[str, UserData], cfg: Config
) -> List[str]:
    nan_frac = compute_feature_nan_fractions(users, cfg)
    max_frac = float(cfg.raw["preprocessing"]["max_feature_nan_fraction"])
    keep = nan_frac[nan_frac <= max_frac].index.tolist()

    exclude_orientation = bool(
        cfg.raw["preprocessing"].get("exclude_orientation_dependent_features", True)
    )
    if exclude_orientation:
        keep = [c for c in keep if not _is_orientation_dependent(c)]

    if not keep:
        raise ValueError(
            "No accelerometer/gyroscope feature columns survived the NaN-fraction "
            f"filter (max_feature_nan_fraction={max_frac}) and orientation-dependent-"
            f"feature exclusion. Check the input data or set "
            f"preprocessing.exclude_orientation_dependent_features to false."
        )
    return sorted(keep)

def _is_orientation_dependent(feature_name: str) -> bool:
    parts = feature_name.split(":")
    if len(parts) < 3 or parts[1] != "3d":
        return False
    subfeature = parts[2]
    return subfeature.startswith("mean_") or subfeature.startswith("ro_")

def _impute_user_features(
    feats: pd.DataFrame, global_median: pd.Series
) -> pd.DataFrame:
    user_median = feats.median(axis=0, skipna=True)
    fill_values = user_median.fillna(global_median)
    return feats.fillna(fill_values)

def add_derived_features(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()

    def _get(col: str) -> Optional[pd.Series]:
        return feats[col] if col in feats.columns else None

    acc_std = _get("raw_acc:magnitude_stats:std")
    gyro_std = _get("proc_gyro:magnitude_stats:std")
    acc_mean = _get("raw_acc:magnitude_stats:mean")
    gyro_mean = _get("proc_gyro:magnitude_stats:mean")

    eps = 1e-6

    if acc_std is not None and gyro_std is not None:
        feats["derived:gyro_to_acc_std_ratio"] = gyro_std / (acc_std + eps)

    if gyro_mean is not None and acc_mean is not None:
        feats["derived:gyro_to_acc_mean_ratio"] = gyro_mean / (acc_mean + eps)

    acc_energy_bands = [
        _get(f"raw_acc:magnitude_spectrum:log_energy_band{i}") for i in range(5)
    ]
    acc_energy_bands = [b for b in acc_energy_bands if b is not None]
    if len(acc_energy_bands) >= 2:
        low = acc_energy_bands[0]
        high = acc_energy_bands[-1]
        feats["derived:acc_spectral_high_low_ratio"] = (high - low) 

    acc_p75 = _get("raw_acc:magnitude_stats:percentile75")
    acc_p25 = _get("raw_acc:magnitude_stats:percentile25")
    if acc_p75 is not None and acc_p25 is not None:
        feats["derived:acc_iqr"] = acc_p75 - acc_p25

    return feats

def prepare_user(
    ud: UserData,
    feature_columns: List[str],
    global_median: pd.Series,
    cfg: Optional[Config] = None,
) -> PreparedUser:
    cfg = cfg or load_config()
    base_feature_columns = [c for c in feature_columns if not c.startswith("derived:")]

    feats = select_modality_features(ud.features, cfg).reindex(columns=base_feature_columns)
    feats = _impute_user_features(feats, global_median.reindex(base_feature_columns))

    if bool(cfg.raw["preprocessing"].get("add_derived_features", True)):
        feats = add_derived_features(feats)

    y = get_primary_activity_label(ud.labels, cfg.activity_classes)

    t_start = int(ud.timestamps.min())
    rel_time_s = (ud.timestamps - t_start).astype(np.int64)

    merge_gap = int(cfg.raw["preprocessing"]["interval_merge_gap_seconds"])
    dt = np.diff(ud.timestamps, prepend=ud.timestamps[0] if len(ud.timestamps) else 0)
    gap_before = dt > merge_gap
    if len(gap_before) > 0:
        gap_before[0] = True  

    return PreparedUser(
        uuid=ud.uuid,
        t_start=t_start,
        rel_time_s=rel_time_s,
        X=feats,
        y=y,
        label_source=ud.label_source,
        gap_before=gap_before,
    )

def prepare_all_users(
    users: Dict[str, UserData], cfg: Optional[Config] = None
) -> Dict[str, PreparedUser]:
    cfg = cfg or load_config()
    feature_columns = select_stable_feature_columns(users, cfg)
    prefixes = tuple(cfg.feature_prefixes.values())
    all_feats = []
    for ud in users.values():
        f = select_modality_features(ud.features, cfg).reindex(columns=feature_columns)
        all_feats.append(f)
    stacked = pd.concat(all_feats, axis=0) if all_feats else pd.DataFrame(columns=feature_columns)
    global_median = stacked.median(axis=0, skipna=True)
    global_median = global_median.fillna(0.0)

    return {
        uuid: prepare_user(ud, feature_columns, global_median, cfg)
        for uuid, ud in users.items()
    }