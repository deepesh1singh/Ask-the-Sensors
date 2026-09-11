from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from ask_the_sensors.config import load_config  # noqa: E402
from ask_the_sensors.data_io import get_primary_activity_label, list_available_uuids, read_all_users  # noqa: E402
from ask_the_sensors.preprocessing import (  # noqa: E402
    compute_feature_nan_fractions,
    resolve_available_activity_classes,
    select_stable_feature_columns,
)

def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()

    uuids = list_available_uuids(cfg)
    if not uuids:
        print("ERROR: no primary-data user files found. Run `python -m scripts.setup_data` first.")
        return 1

    print(f"Reading {len(uuids)} users...")
    users = read_all_users(uuids, cfg)

    nan_fracs = compute_feature_nan_fractions(users, cfg)
    feature_columns = select_stable_feature_columns(users, cfg)
    print(f"\n{len(nan_fracs)} accel/gyro feature columns found; "
          f"{len(feature_columns)} kept after NaN-fraction filter "
          f"(threshold={cfg.raw['preprocessing']['max_feature_nan_fraction']}).")

    available_classes, unavailable_classes = resolve_available_activity_classes(users, cfg)
    if unavailable_classes:
        print(
            f"\nWARNING: the following configured activity class(es) have NO usable "
            f"ground-truth examples (missing label column, or present but always "
            f"0/NaN) across all {len(uuids)} users in this dataset copy:\n"
            f"  {unavailable_classes}\n"
            f"They will show a count of 0 below, and scripts.train / scripts.evaluate "
            f"will automatically exclude them from training and evaluation. See "
            f"preprocessing.py::resolve_available_activity_classes for why this is "
            f"checked against the real files rather than assumed from the brief or "
            f"from ExtraSensory's own published aggregate statistics, which can differ "
            f"from a specific downloaded copy of the dataset."
        )

    print(f"\nLabel distribution across all users ({len(cfg.activity_classes)}-class primary activity):")
    total_counts = pd.Series(0, index=cfg.activity_classes + ["<unlabeled>"], dtype=int)
    for ud in users.values():
        primary = get_primary_activity_label(ud.labels, cfg.activity_classes)
        counts = primary.value_counts(dropna=False)
        for cls in cfg.activity_classes:
            total_counts[cls] += int(counts.get(cls, 0))
        n_nan = int(primary.isna().sum())
        total_counts["<unlabeled>"] += n_nan

    for cls, count in total_counts.items():
        display = cfg.activity_display_names.get(cls, cls)
        flag = "  <-- UNAVAILABLE in this dataset copy" if cls in unavailable_classes else ""
        print(f"  {display:30s} {count:8d}{flag}")

    summary = {
        "n_users": len(uuids),
        "n_accel_gyro_feature_columns_total": int(len(nan_fracs)),
        "n_feature_columns_kept": int(len(feature_columns)),
        "feature_columns_kept": feature_columns,
        "label_counts": total_counts.to_dict(),
        "available_activity_classes": available_classes,
        "unavailable_activity_classes": unavailable_classes,
    }
    out_path = cfg.logs_dir / "feature_inspection_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())