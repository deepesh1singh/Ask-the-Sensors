from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from tqdm import tqdm
from ask_the_sensors.config import load_config  
from ask_the_sensors.cv_split import load_cv_folds, restrict_folds_to_available 
from ask_the_sensors.data_io import list_available_uuids, read_all_users  
from ask_the_sensors.preprocessing import (  
    compute_feature_nan_fractions,
    prepare_user,
    resolve_available_activity_classes,
    select_stable_feature_columns,
)
from ask_the_sensors.recognition_model import (  
    model_param_count,
    model_size_on_disk_bytes,
    predict,
    predict_proba,
    save_model,
    train_model,
)

def _stack_fold_training_data(
    users, uuids: List[str], feature_columns: List[str], global_median: pd.Series, cfg
):
    X_list, y_list, meta_list = [], [], []
    for uuid in uuids:
        ud = users[uuid]
        prep = prepare_user(ud, feature_columns, global_median, cfg)
        mask = prep.y.notna()
        if mask.sum() == 0:
            continue
        X_list.append(prep.X.loc[mask])
        y_list.append(prep.y.loc[mask])
        meta_list.append(
            pd.DataFrame(
                {
                    "uuid": uuid,
                    "timestamp": prep.X.loc[mask].index.to_numpy(),
                    "rel_time_s": prep.rel_time_s[mask.to_numpy()],
                }
            )
        )
    if not X_list:
        return None, None, None
    X = pd.concat(X_list, axis=0).reset_index(drop=True)
    y = pd.concat(y_list, axis=0).reset_index(drop=True)
    meta = pd.concat(meta_list, axis=0).reset_index(drop=True)
    return X, y, meta

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Fold indices to run (default: all)")
    parser.add_argument(
        "--models", type=str, nargs="*", default=["full", "hierarchical", "compressed"],
        choices=["full", "hierarchical", "compressed"],
        help="Which recognition-model variants to train. 'full': flat "
             "gradient-boosted-tree classifier. 'hierarchical': two-stage "
             "stationary/moving classifier (see recognition_model.py); "
             "typically the most accurate of the three on this project's "
             "real evaluation results. 'compressed': small MLP for the "
             "accuracy-vs-overhead edge comparison.",
    )
    args = parser.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()

    np.random.seed(cfg.seed)

    print("Loading available users...")
    uuids = list_available_uuids(cfg)
    if len(uuids) == 0:
        print(
            "ERROR: no primary-data user files found. "
            "Run `python -m scripts.setup_data` first (see README.md)."
        )
        return 1
    print(f"  {len(uuids)} users available in {cfg.primary_data_dir}")

    print("Loading CV folds...")
    folds = load_cv_folds(cfg)
    folds = restrict_folds_to_available(folds, set(uuids))

    fold_indices = args.folds if args.folds else [f.fold_index for f in folds]
    folds = [f for f in folds if f.fold_index in fold_indices]

    print("Reading all user data into memory (this can take a minute)...")
    t0 = time.time()
    users = read_all_users(uuids, cfg)
    print(f"  loaded {len(users)} users in {time.time() - t0:.1f}s")

    available_classes, unavailable_classes = resolve_available_activity_classes(users, cfg)
    if unavailable_classes:
        print(
            f"\nWARNING: the following requested activity class(es) have NO usable "
            f"ground-truth examples in this dataset copy and will be EXCLUDED from "
            f"training and evaluation: {unavailable_classes}\n"
            f"  (Requested via configs/config.yaml: activity_classes; "
            f"actually usable here: {available_classes})\n"
        )
    cfg.raw["activity_classes"] = available_classes

    all_oof_predictions = []
    fold_feature_columns: Dict[int, List[str]] = {}
    training_summary = []

    for fold in folds:
        print(f"\n=== Fold {fold.fold_index}: train={len(fold.train_uuids)} users, "
              f"test={len(fold.test_uuids)} users ===")

        train_users = {u: users[u] for u in fold.train_uuids if u in users}
        test_users = {u: users[u] for u in fold.test_uuids if u in users}

        if not train_users or not test_users:
            print(f"  Skipping fold {fold.fold_index}: empty train or test set after restriction.")
            continue

        feature_columns = select_stable_feature_columns(train_users, cfg)
        fold_feature_columns[fold.fold_index] = feature_columns

        prefixes = tuple(cfg.feature_prefixes.values())
        train_feats_concat = pd.concat(
            [
                train_users[u].features[
                    [c for c in train_users[u].features.columns if c.split(":")[0] in prefixes]
                ].reindex(columns=feature_columns)
                for u in train_users
            ],
            axis=0,
        )
        global_median = train_feats_concat.median(axis=0, skipna=True).fillna(0.0)

        X_train, y_train, _ = _stack_fold_training_data(
            train_users, list(train_users.keys()), feature_columns, global_median, cfg
        )
        if X_train is None:
            print(f"  Skipping fold {fold.fold_index}: no labelled training examples.")
            continue

        print(f"  Training examples: {len(X_train)}; class distribution:")
        print(f"  {y_train.value_counts().to_dict()}")

        for model_name in args.models:
            t0 = time.time()
            model = train_model(model_name, X_train, y_train, cfg)
            train_time_s = time.time() - t0

            model_path = cfg.models_dir / f"fold_{fold.fold_index}_{model_name}.joblib"
            save_model(model, model_path)

            size_mb = model_size_on_disk_bytes(model) / (1024 * 1024)
            n_params = model_param_count(model)
            print(f"  [{model_name}] trained in {train_time_s:.1f}s, "
                  f"size={size_mb:.3f} MB, params={n_params}")

            X_test, y_test, meta_test = _stack_fold_training_data(
                test_users, list(test_users.keys()), feature_columns, global_median, cfg
            )
            if X_test is None:
                print(f"  No labelled test examples for fold {fold.fold_index}; skipping predictions.")
                continue

            y_pred = predict(model, X_test)
            proba = predict_proba(model, X_test)
            confidence = proba.max(axis=1).to_numpy()

            fold_result = meta_test.copy()
            fold_result["fold"] = fold.fold_index
            fold_result["model"] = model_name
            fold_result["y_true"] = y_test.to_numpy()
            fold_result["y_pred"] = y_pred
            fold_result["confidence"] = confidence
            all_oof_predictions.append(fold_result)

            training_summary.append(
                {
                    "fold": fold.fold_index,
                    "model": model_name,
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "train_time_s": round(train_time_s, 3),
                    "size_mb": round(size_mb, 4),
                    "param_count": n_params,
                }
            )

    if not all_oof_predictions:
        print("ERROR: no out-of-fold predictions were produced. Check data availability.")
        return 1

    oof_df = pd.concat(all_oof_predictions, axis=0).reset_index(drop=True)
    oof_path = cfg.logs_dir / "cv_predictions.csv.gz"
    oof_df.to_csv(oof_path, index=False, compression="gzip")
    print(f"\nSaved out-of-fold predictions ({len(oof_df)} rows) to {oof_path}")

    feat_cols_path = cfg.logs_dir / "fold_feature_columns.json"
    with open(feat_cols_path, "w") as f:
        json.dump(fold_feature_columns, f, indent=2)
    print(f"Saved per-fold feature column lists to {feat_cols_path}")

    summary_path = cfg.logs_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"Saved training summary to {summary_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())