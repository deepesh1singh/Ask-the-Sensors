from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from ask_the_sensors.aggregation import build_timeline, total_duration_for_activity, compare_durations  
from ask_the_sensors.config import load_config  
from ask_the_sensors.cv_split import load_cv_folds, restrict_folds_to_available  
from ask_the_sensors.data_io import list_available_uuids, read_all_users  
from ask_the_sensors.efficiency import measure_model_efficiency, report_to_dict  
from ask_the_sensors.figures import (  
    figure_accuracy_by_question_type,
    figure_accuracy_vs_overhead,
    figure_accuracy_vs_strictness,
    figure_confusion_matrix,
    figure_robustness_curve,
)
from ask_the_sensors.metrics import (  
    accuracy_vs_iou_curve,
    accuracy_vs_numeric_tolerance_curve,
    binary_verification_metrics,
    categorical_accuracy,
    categorical_balanced_accuracy,
    categorical_macro_f1,
    compute_confusion_matrix,
    interval_iou,
    macro_average_qa_accuracy,
    mean_absolute_error,
    numeric_accuracy_within_tolerance,
    per_class_prf1,
)
from ask_the_sensors.preprocessing import add_derived_features, prepare_user, select_stable_feature_columns 
from ask_the_sensors.recognition_model import load_model, predict

def _load_oof_predictions(cfg) -> pd.DataFrame:
    path = cfg.logs_dir / "cv_predictions.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m scripts.train` before `python -m scripts.evaluate`."
        )
    return pd.read_csv(path, compression="gzip")

def _headline_model_name(oof_df: pd.DataFrame, cfg) -> str:
    requested = str(cfg.raw["evaluation"].get("headline_model", "hierarchical"))
    available = set(oof_df["model"].unique().tolist())
    if requested in available:
        return requested

    for fallback in ("hierarchical", "full"):
        if fallback in available:
            print(
                f"NOTE: configured headline_model={requested!r} not found in "
                f"out-of-fold predictions (models present: {sorted(available)}); "
                f"using {fallback!r} for headline metrics instead."
            )
            return fallback

    fallback = sorted(available)[0]
    print(
        f"NOTE: configured headline_model={requested!r} not found; using "
        f"{fallback!r} (first available model) for headline metrics instead."
    )
    return fallback

def evaluate_identification_and_verification(oof_df: pd.DataFrame, cfg, model_name: str) -> Dict[str, float]:
    full_df = oof_df[oof_df["model"] == model_name]
    labels = cfg.activity_classes

    identify_acc = categorical_accuracy(full_df["y_true"], full_df["y_pred"])

    verify_f1s = []
    for label in labels:
        y_true_bin = (full_df["y_true"] == label).to_numpy()
        y_pred_bin = (full_df["y_pred"] == label).to_numpy()
        m = binary_verification_metrics(y_true_bin, y_pred_bin)
        if m.f1 == m.f1:
            verify_f1s.append(m.f1)
    verify_acc = float(np.mean(verify_f1s)) if verify_f1s else float("nan")

    return {"identification": identify_acc, "verification": verify_acc}

def evaluate_duration_and_count(oof_df: pd.DataFrame, cfg, model_name: str) -> Dict[str, float]:
    full_df = oof_df[oof_df["model"] == model_name].copy()

    dur_true, dur_pred = [], []
    count_true, count_pred = [], []

    for (uuid, fold), group in full_df.groupby(["uuid", "fold"]):
        group = group.sort_values("rel_time_s")
        rel_time = group["rel_time_s"].to_numpy()
        gap_before = np.zeros(len(rel_time), dtype=bool)
        if len(rel_time) > 0:
            dt = np.diff(rel_time, prepend=rel_time[0])
            merge_gap = int(cfg.raw["preprocessing"]["interval_merge_gap_seconds"])
            gap_before = dt > merge_gap
            gap_before[0] = True

        tl_true = build_timeline(uuid, rel_time, group["y_true"].to_numpy(), gap_before=gap_before, cfg=cfg)
        tl_pred = build_timeline(uuid, rel_time, group["y_pred"].to_numpy(), gap_before=gap_before, cfg=cfg)

        for activity in cfg.activity_classes:
            dt_true, _ = total_duration_for_activity(tl_true, activity)
            dt_pred, _ = total_duration_for_activity(tl_pred, activity)
            dur_true.append(dt_true)
            dur_pred.append(dt_pred)

            ivs_true = [iv for iv in tl_true.intervals if iv.activity == activity]
            ivs_pred = [iv for iv in tl_pred.intervals if iv.activity == activity]
            count_true.append(len(ivs_true))
            count_pred.append(len(ivs_pred))

    eval_cfg = cfg.raw["evaluation"]
    duration_acc = numeric_accuracy_within_tolerance(
        dur_true, dur_pred,
        abs_tolerance=float(eval_cfg["numeric_tolerance_seconds"]),
        rel_tolerance=float(eval_cfg["numeric_relative_tolerance"]),
    )
    count_acc = numeric_accuracy_within_tolerance(count_true, count_pred, abs_tolerance=1.0)

    duration_mae = mean_absolute_error(dur_true, dur_pred)
    count_mae = mean_absolute_error(count_true, count_pred)

    return {
        "duration": duration_acc,
        "count": count_acc,
        "_duration_mae_seconds": duration_mae,
        "_count_mae": count_mae,
        "_duration_pairs": (dur_true, dur_pred),
    }

def evaluate_comparison(oof_df: pd.DataFrame, cfg, model_name: str) -> float:
    full_df = oof_df[oof_df["model"] == model_name].copy()
    correct, total = 0, 0
    classes = cfg.activity_classes

    for (uuid, fold), group in full_df.groupby(["uuid", "fold"]):
        group = group.sort_values("rel_time_s")
        rel_time = group["rel_time_s"].to_numpy()
        tl_true = build_timeline(uuid, rel_time, group["y_true"].to_numpy(), cfg=cfg)
        tl_pred = build_timeline(uuid, rel_time, group["y_pred"].to_numpy(), cfg=cfg)

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                a, b = classes[i], classes[j]
                winner_true, dt_a_true, dt_b_true, _, _ = compare_durations(tl_true, a, b)
                winner_pred, _, _, _, _ = compare_durations(tl_pred, a, b)
                if dt_a_true == 0 and dt_b_true == 0:
                    continue  
                total += 1
                if winner_true == winner_pred:
                    correct += 1

    return correct / total if total > 0 else float("nan")

def evaluate_grounding_iou(oof_df: pd.DataFrame, cfg, model_name: str) -> List[float]:
    full_df = oof_df[oof_df["model"] == model_name].copy()
    ious = []

    for (uuid, fold), group in full_df.groupby(["uuid", "fold"]):
        group = group.sort_values("rel_time_s")
        rel_time = group["rel_time_s"].to_numpy()
        tl_true = build_timeline(uuid, rel_time, group["y_true"].to_numpy(), cfg=cfg)
        tl_pred = build_timeline(uuid, rel_time, group["y_pred"].to_numpy(), cfg=cfg)

        for true_iv in tl_true.intervals:
            candidates = [p for p in tl_pred.intervals if p.activity == true_iv.activity]
            if not candidates:
                ious.append(0.0)
                continue
            best_iou = max(
                interval_iou(c.start_s, c.end_s, true_iv.start_s, true_iv.end_s) for c in candidates
            )
            ious.append(best_iou)

    return ious

def evaluate_robustness(oof_df: pd.DataFrame, cfg) -> Dict[str, List[float]]:
    cfg_eval = cfg.raw["evaluation"]["robustness"]
    noise_fracs = cfg_eval["noise_std_fractions"]
    dropout_fracs = cfg_eval["dropout_fractions"]

    uuids = list_available_uuids(cfg)
    users = read_all_users(uuids, cfg)
    folds = restrict_folds_to_available(load_cv_folds(cfg), set(uuids))

    rng = np.random.RandomState(cfg.seed)

    noise_accs, dropout_accs = [], []

    fold0 = next((f for f in folds if f.fold_index == 0), folds[0])
    headline_model = str(cfg.raw["evaluation"].get("headline_model", "hierarchical"))
    model_path = None
    for model_name in (headline_model, "hierarchical", "full"):
        candidate = cfg.models_dir / f"fold_0_{model_name}.joblib"
        if candidate.is_file():
            model_path = candidate
            break
    if model_path is None:
        print("  (skipping robustness curve: no fold_0_hierarchical.joblib or fold_0_full.joblib found)")
        return {"noise_std_fraction": [], "noise_accuracy": [], "dropout_fraction": [], "dropout_accuracy": []}

    model = load_model(model_path)
    feature_columns = model.feature_columns

    test_users = {u: users[u] for u in fold0.test_uuids if u in users}
    train_users = {u: users[u] for u in fold0.train_uuids if u in users}
    prefixes = tuple(cfg.feature_prefixes.values())
    base_feature_columns = [c for c in feature_columns if not c.startswith("derived:")]
    train_feats_concat = pd.concat(
        [
            train_users[u].features[
                [c for c in train_users[u].features.columns if c.split(":")[0] in prefixes]
            ].reindex(columns=base_feature_columns)
            for u in train_users
        ],
        axis=0,
    )
    global_median = train_feats_concat.median(axis=0, skipna=True).fillna(0.0)
    feature_std = train_feats_concat.std(axis=0, skipna=True).fillna(0.0)

    X_list, y_list = [], []
    for uuid, ud in test_users.items():
        prep = prepare_user(ud, feature_columns, global_median, cfg)
        mask = prep.y.notna()
        if mask.sum() == 0:
            continue
        X_list.append(prep.X.loc[mask])
        y_list.append(prep.y.loc[mask])
    if not X_list:
        return {"noise_std_fraction": [], "noise_accuracy": [], "dropout_fraction": [], "dropout_accuracy": []}

    X_test = pd.concat(X_list, axis=0)
    y_test = pd.concat(y_list, axis=0)

    def _recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
        if bool(cfg.raw["preprocessing"].get("add_derived_features", True)):
            base_only = df[[c for c in df.columns if not c.startswith("derived:")]]
            return add_derived_features(base_only).reindex(columns=feature_columns)
        return df

    for frac in noise_fracs:
        X_noisy = X_test.copy()
        if frac > 0:
            base_cols_present = [c for c in base_feature_columns if c in X_noisy.columns]
            noise = rng.normal(0.0, frac, size=(len(X_noisy), len(base_cols_present)))
            noise = noise * feature_std[base_cols_present].to_numpy()
            X_noisy[base_cols_present] = X_noisy[base_cols_present] + noise
            X_noisy = _recompute_derived(X_noisy)
        y_pred = predict(model, X_noisy)
        noise_accs.append(categorical_accuracy(y_test.to_numpy(), y_pred))

    for frac in dropout_fracs:
        X_dropped = X_test.copy()
        if frac > 0:
            base_cols_present = [c for c in base_feature_columns if c in X_dropped.columns]
            drop_mask = rng.random((len(X_dropped), len(base_cols_present))) < frac
            base_block = X_dropped[base_cols_present].mask(drop_mask, other=np.nan)
            base_block = base_block.fillna(global_median[base_cols_present])
            X_dropped[base_cols_present] = base_block
            X_dropped = _recompute_derived(X_dropped)
        y_pred = predict(model, X_dropped)
        dropout_accs.append(categorical_accuracy(y_test.to_numpy(), y_pred))

    return {
        "noise_std_fraction": list(noise_fracs),
        "noise_accuracy": noise_accs,
        "dropout_fraction": list(dropout_fracs),
        "dropout_accuracy": dropout_accs,
    }

def evaluate_efficiency(cfg) -> List[Dict]:
    reports = []
    for model_name in ["full", "hierarchical", "compressed"]:
        model_path = cfg.models_dir / f"fold_0_{model_name}.joblib"
        if not model_path.is_file():
            print(f"  (skipping efficiency measurement: {model_path} not found)")
            continue
        model = load_model(model_path)

        uuids = list_available_uuids(cfg)
        users = read_all_users(uuids[:1], cfg)  # one user is enough for a timing sample
        u = uuids[0]
        prep = prepare_user(users[u], model.feature_columns, pd.Series(0.0, index=model.feature_columns), cfg)
        X_single = prep.X.iloc[[0]]
        X_batch = prep.X

        report = measure_model_efficiency(model, X_single, X_batch, cfg)
        reports.append(report_to_dict(report))
    return reports

def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()

    print("Loading out-of-fold predictions...")
    oof_df = _load_oof_predictions(cfg)
    observed_classes = set(oof_df["y_true"].dropna().unique().tolist())
    resolved_classes = [c for c in cfg.activity_classes if c in observed_classes]
    dropped_classes = [c for c in cfg.activity_classes if c not in observed_classes]
    if dropped_classes:
        print(
            f"NOTE: the following configured activity class(es) do not appear in the "
            f"out-of-fold predictions (no ground-truth examples were available when "
            f"scripts.train ran) and will be excluded from this evaluation's confusion "
            f"matrix and per-class metrics: {dropped_classes}"
        )
    cfg.raw["activity_classes"] = resolved_classes

    headline_model = _headline_model_name(oof_df, cfg)
    print(f"Using '{headline_model}' model's out-of-fold predictions for headline metrics "
          f"(Figures 1, 2, 3, 5). See evaluation.headline_model in configs/config.yaml.")

    results: Dict[str, object] = {}
    results["headline_model"] = headline_model

    print("Evaluating identification / verification (Tier 1)...")
    tier1 = evaluate_identification_and_verification(oof_df, cfg, headline_model)
    results["tier1"] = tier1

    print("Evaluating duration / count (Tier 2)...")
    tier2_dc = evaluate_duration_and_count(oof_df, cfg, headline_model)
    duration_pairs = tier2_dc.pop("_duration_pairs")
    results["tier2_duration_count"] = tier2_dc

    print("Evaluating comparison (Tier 2)...")
    comparison_acc = evaluate_comparison(oof_df, cfg, headline_model)
    results["tier2_comparison"] = comparison_acc

    print("Evaluating evidence grounding IoU (Tier 3)...")
    ious = evaluate_grounding_iou(oof_df, cfg, headline_model)
    grounding_acc = float(np.mean([iou >= cfg.raw["evaluation"]["default_iou_threshold"] for iou in ious])) if ious else float("nan")
    results["tier3_grounding_accuracy"] = grounding_acc
    results["tier3_mean_iou"] = float(np.mean(ious)) if ious else float("nan")

    # ---- Figure 1: accuracy by question type ------------------------------
    full_df = oof_df[oof_df["model"] == headline_model]
    labels = cfg.activity_classes
    per_type_accuracy = {
        "identification": tier1["identification"],
        "verification": tier1["verification"],
        "duration": tier2_dc["duration"],
        "count": tier2_dc["count"],
        "comparison": comparison_acc,
        "grounding": grounding_acc,
    }
    overall_acc = macro_average_qa_accuracy(per_type_accuracy)
    results["overall_qa_accuracy"] = overall_acc

    correctness_rules = {
        "identification": "exact match",
        "verification": "F1 on positive class (avg over 7 classes)",
        "duration": f"within {cfg.raw['evaluation']['numeric_tolerance_seconds']}s abs or "
                    f"{cfg.raw['evaluation']['numeric_relative_tolerance']*100:.0f}% rel",
        "count": "within +/-1",
        "comparison": "exact match on which activity had more total time",
        "grounding": f"IoU >= {cfg.raw['evaluation']['default_iou_threshold']}",
    }
    fig1_path = cfg.figures_dir / "figure1_accuracy_by_question_type.png"
    figure_accuracy_by_question_type(per_type_accuracy, overall_acc, correctness_rules, fig1_path)
    print(f"  Saved {fig1_path}")

    # ---- Figure 2: confusion matrix -----------------------------------------
    cm = compute_confusion_matrix(full_df["y_true"], full_df["y_pred"], labels)
    per_class = per_class_prf1(full_df["y_true"], full_df["y_pred"], labels)
    fig2_path = cfg.figures_dir / "figure2_confusion_matrix.png"
    figure_confusion_matrix(cm, labels, per_class, fig2_path)
    print(f"  Saved {fig2_path}")
    results["confusion_matrix"] = cm.tolist()
    results["per_class_prf1"] = per_class
    results["macro_f1"] = categorical_macro_f1(full_df["y_true"], full_df["y_pred"], labels)
    results["balanced_accuracy"] = categorical_balanced_accuracy(full_df["y_true"], full_df["y_pred"])

    # ---- Figure 3: accuracy vs strictness -----------------------------------
    tolerances = list(range(0, 601, 30))
    numeric_curve = accuracy_vs_numeric_tolerance_curve(duration_pairs[0], duration_pairs[1], tolerances)
    iou_thresholds = cfg.raw["evaluation"]["iou_thresholds"]
    iou_curve = accuracy_vs_iou_curve(ious, iou_thresholds) if ious else {}
    fig3_path = cfg.figures_dir / "figure3_accuracy_vs_strictness.png"
    figure_accuracy_vs_strictness(numeric_curve, iou_curve, fig3_path)
    print(f"  Saved {fig3_path}")
    results["numeric_tolerance_curve"] = numeric_curve
    results["iou_curve"] = iou_curve

    # ---- Figure 4: accuracy vs overhead -------------------------------------
    print("Measuring model efficiency (full vs hierarchical vs compressed)...")
    eff_reports = evaluate_efficiency(cfg)
    results["efficiency_reports"] = eff_reports
    if eff_reports:
        points = []
        for model_name in ["full", "hierarchical", "compressed"]:
            model_df = oof_df[oof_df["model"] == model_name]
            if len(model_df) == 0:
                continue
            acc = categorical_accuracy(model_df["y_true"], model_df["y_pred"])
            rep = next((r for r in eff_reports if r["model_name"] == model_name), None)
            if rep is None:
                continue
            points.append(
                {
                    "name": model_name,
                    "accuracy": acc,
                    "size_mb": rep["size_on_disk_mb"],
                    "latency_ms": rep["mean_latency_ms"],
                    "memory_mb": rep["peak_memory_mb"],
                }
            )
        if points:
            fig4_path = cfg.figures_dir / "figure4_accuracy_vs_overhead.png"
            figure_accuracy_vs_overhead(points, "size_mb", "Model size on disk (MB)", fig4_path)
            print(f"  Saved {fig4_path}")
            results["overhead_points"] = points

    # ---- Figure 5: robustness curve -----------------------------------------
    print("Running robustness experiments (noise + dropout injection)...")
    robustness = evaluate_robustness(oof_df, cfg)
    if robustness["noise_accuracy"]:
        fig5a_path = cfg.figures_dir / "figure5a_robustness_noise.png"
        figure_robustness_curve(
            robustness["noise_std_fraction"], robustness["noise_accuracy"],
            "Injected noise (fraction of per-feature std)", fig5a_path,
        )
        print(f"  Saved {fig5a_path}")
    if robustness["dropout_accuracy"]:
        fig5b_path = cfg.figures_dir / "figure5b_robustness_dropout.png"
        figure_robustness_curve(
            robustness["dropout_fraction"], robustness["dropout_accuracy"],
            "Fraction of features dropped (set to NaN then imputed)", fig5b_path,
        )
        print(f"  Saved {fig5b_path}")
    results["robustness"] = robustness

    # ---- Persist everything --------------------------------------------------
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    results_path = cfg.logs_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_default)
    print(f"\nSaved full evaluation results to {results_path}")

    print("\n=== Summary ===")
    print(f"Headline model: {headline_model!r} (set via evaluation.headline_model)")
    print(f"Overall macro-averaged QA accuracy: {overall_acc:.3f}")
    for k, v in per_type_accuracy.items():
        print(f"  {k}: {v:.3f}")
    print(f"Macro-F1 (7-class recognition): {results['macro_f1']:.3f}")
    print(f"Mean grounding IoU: {results['tier3_mean_iou']:.3f}")
    if "overhead_points" in results:
        print("\nAll trained models' accuracy (for comparison, see Figure 4):")
        for point in results["overhead_points"]:
            marker = " <- headline model" if point["name"] == headline_model else ""
            print(f"  {point['name']:14s} accuracy={point['accuracy']:.3f}{marker}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())