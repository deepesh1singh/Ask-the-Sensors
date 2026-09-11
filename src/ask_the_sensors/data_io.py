from __future__ import annotations
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .config import Config, load_config

UUID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)

@dataclass
class UserData:
    uuid: str
    timestamps: np.ndarray         
    features: pd.DataFrame         
    labels: pd.DataFrame          
    label_source: np.ndarray      

def list_available_uuids(cfg: Optional[Config] = None) -> List[str]:
    cfg = cfg or load_config()
    data_dir = cfg.primary_data_dir
    if not data_dir.is_dir():
        return []
    uuids = set()
    for pattern in ("*.features_labels.csv.gz", "*.features_labels.csv"):
        for p in sorted(data_dir.glob(pattern)):
            uuid = p.name.split(".")[0]
            if UUID_RE.match(uuid):
                uuids.add(uuid)
    return sorted(uuids)

def _user_file_path(uuid: str, cfg: Config) -> Path:
    gz_path = cfg.primary_data_dir / f"{uuid}.features_labels.csv.gz"
    if gz_path.is_file():
        return gz_path
    return cfg.primary_data_dir / f"{uuid}.features_labels.csv"

def _parse_header(columns: List[str]) -> Tuple[List[str], List[str]]:
    assert columns[0] == "timestamp", f"Expected first column 'timestamp', got {columns[0]!r}"
    assert columns[-1] == "label_source", f"Expected last column 'label_source', got {columns[-1]!r}"

    first_label_idx = None
    for i, col in enumerate(columns):
        if col.startswith("label:"):
            first_label_idx = i
            break
    if first_label_idx is None:
        raise ValueError("No 'label:' columns found in header; unexpected file format.")

    feature_names = columns[1:first_label_idx]
    label_names_raw = columns[first_label_idx:-1]
    label_names = [c[len("label:"):] for c in label_names_raw]
    return feature_names, label_names

def read_user_data(uuid: str, cfg: Optional[Config] = None) -> UserData:
    cfg = cfg or load_config()
    fpath = _user_file_path(uuid, cfg)
    if not fpath.is_file():
        raise FileNotFoundError(
            f"Primary data file not found for user {uuid}: {fpath}\n"
            f"Did you run `python -m scripts.setup_data`? See README.md section 2."
        )

    if fpath.suffix == ".gz":
        with gzip.open(fpath, "rt") as f:
            header_line = f.readline()
    else:
        with open(fpath, "rt") as f:
            header_line = f.readline()
    all_columns = header_line.rstrip("\n").rstrip("\r").split(",")

    feature_names_all, label_names = _parse_header(all_columns)
    prefixes = tuple(cfg.feature_prefixes.values())
    kept_feature_names = [c for c in feature_names_all if c.split(":")[0] in prefixes]
    label_columns = [f"label:{name}" for name in label_names]
    usecols = ["timestamp"] + kept_feature_names + label_columns + ["label_source"]

    if fpath.suffix == ".gz":
        with gzip.open(fpath, "rt") as f:
            df = pd.read_csv(f, usecols=usecols, low_memory=False)
    else:
        df = pd.read_csv(fpath, usecols=usecols, low_memory=False)

    df = df[usecols]
    columns = list(df.columns)
    feature_names, label_names = _parse_header(columns)

    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]

    features = df[feature_names].to_numpy(dtype=np.float64)[order]
    features_df = pd.DataFrame(features, index=timestamps, columns=feature_names)

    label_cols = [f"label:{name}" for name in label_names]
    labels = df[label_cols].to_numpy(dtype=np.float64)[order]
    labels_df = pd.DataFrame(labels, index=timestamps, columns=label_names)

    label_source = df["label_source"].to_numpy(dtype=np.int64)[order]

    return UserData(
        uuid=uuid,
        timestamps=timestamps,
        features=features_df,
        labels=labels_df,
        label_source=label_source,
    )

def select_modality_features(features_df: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    prefixes = tuple(cfg.feature_prefixes.values())
    keep_cols = [c for c in features_df.columns if c.split(":")[0] in prefixes]
    if not keep_cols:
        raise ValueError(
            f"No columns matched accelerometer/gyroscope prefixes {prefixes}. "
            f"Available prefixes: {sorted(set(c.split(':')[0] for c in features_df.columns))}"
        )
    return features_df[keep_cols]


def get_primary_activity_label(
    labels_df: pd.DataFrame,
    activity_classes: List[str],
) -> pd.Series:
    available_classes = [c for c in activity_classes if c in labels_df.columns]
    sub = labels_df[available_classes] if available_classes else pd.DataFrame(index=labels_df.index)
    is_one = (sub == 1.0)
    count_one = is_one.sum(axis=1) if available_classes else pd.Series(0, index=labels_df.index)

    result = pd.Series(index=labels_df.index, dtype=object)
    result[:] = np.nan

    unique_mask = count_one == 1
    if unique_mask.any():
        sub_unique = is_one.loc[unique_mask]
        chosen = sub_unique.idxmax(axis=1)
        result.loc[unique_mask] = chosen.values

    return result

def read_all_users(
    uuids: Optional[List[str]] = None, cfg: Optional[Config] = None
) -> Dict[str, UserData]:
    cfg = cfg or load_config()
    uuids = uuids if uuids is not None else list_available_uuids(cfg)
    return {u: read_user_data(u, cfg) for u in uuids}

if __name__ == "__main__":
    cfg = load_config()
    uuids = list_available_uuids(cfg)
    print(f"Found {len(uuids)} user files in {cfg.primary_data_dir}")
    if uuids:
        ud = read_user_data(uuids[0], cfg)
        acc_gyro = select_modality_features(ud.features, cfg)
        print(f"User {ud.uuid}: {len(ud.timestamps)} examples, "
              f"{acc_gyro.shape[1]} accel+gyro features")
        primary = get_primary_activity_label(ud.labels, cfg.activity_classes)
        print(primary.value_counts(dropna=False))