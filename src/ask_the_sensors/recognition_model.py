from __future__ import annotations
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .config import Config, load_config

@dataclass
class TrainedModel:
    name: str                      
    classifier: object             
    scaler: StandardScaler
    classes_: List[str]
    feature_columns: List[str]
    quantize_dtype: Optional[str] = None

STATIONARY_GROUP = ["LYING_DOWN", "SITTING", "OR_standing", "STANDING_AND_MOVING"]
MOVING_GROUP = ["FIX_walking", "FIX_running", "BICYCLING"]

def _group_for_class(cls: str) -> str:
    if cls in STATIONARY_GROUP:
        return "stationary"
    if cls in MOVING_GROUP:
        return "moving"
    raise ValueError(f"Unrecognized activity class for hierarchical grouping: {cls!r}")

def _build_full_classifier(cfg: Config) -> HistGradientBoostingClassifier:
    mc = cfg.raw["model"]["full"]
    return HistGradientBoostingClassifier(
        max_iter=int(mc["n_estimators"]),
        max_depth=int(mc["max_depth"]),
        max_leaf_nodes=int(mc.get("max_leaf_nodes", 63)),
        learning_rate=float(mc["learning_rate"]),
        random_state=int(mc["random_state"]),
        early_stopping=False,
        l2_regularization=float(mc.get("l2_regularization", 0.1)),
        class_weight="balanced",
    )

def _build_full_classifier_stage(cfg: Config, stage: str) -> HistGradientBoostingClassifier:
    mc = dict(cfg.raw["model"]["full"])
    mc.update(cfg.raw.get("model", {}).get("hierarchical", {}).get(stage, {}))
    return HistGradientBoostingClassifier(
        max_iter=int(mc["n_estimators"]),
        max_depth=int(mc["max_depth"]),
        max_leaf_nodes=int(mc.get("max_leaf_nodes", 63)),
        learning_rate=float(mc["learning_rate"]),
        random_state=int(mc["random_state"]),
        early_stopping=False,
        l2_regularization=float(mc.get("l2_regularization", 0.1)),
        class_weight="balanced",
    )

def _smote_oversample(
    X: np.ndarray, y: np.ndarray, seed: int, k_neighbors: int = 5, min_class_size: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    target_count = int(counts.max())
    smote_targets = {
        c: target_count for c, n in zip(classes, counts) if min_class_size <= n < target_count
    }
    if not smote_targets:
        return X, y

    effective_k = min(k_neighbors, min(counts[np.isin(classes, list(smote_targets))]) - 1)
    effective_k = max(1, effective_k)

    smote = SMOTE(
        sampling_strategy=smote_targets,
        k_neighbors=effective_k,
        random_state=seed,
    )
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def _build_compressed_classifier(cfg: Config) -> MLPClassifier:
    mc = cfg.raw["model"]["compressed"]
    return MLPClassifier(
        hidden_layer_sizes=tuple(mc["hidden_layer_sizes"]),
        alpha=float(mc["alpha"]),
        max_iter=int(mc["max_iter"]),
        random_state=int(mc["random_state"]),
        early_stopping=False,
    )

def _quantize_mlp_weights(clf: MLPClassifier, dtype: str) -> MLPClassifier:
    np_dtype = np.dtype(dtype)
    for i in range(len(clf.coefs_)):
        clf.coefs_[i] = clf.coefs_[i].astype(np_dtype)
    for i in range(len(clf.intercepts_)):
        clf.intercepts_[i] = clf.intercepts_[i].astype(np_dtype)
    return clf

def train_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Optional[Config] = None,
) -> TrainedModel:
    cfg = cfg or load_config()
    assert name in ("full", "hierarchical", "compressed"), name
    assert not y_train.isna().any(), "y_train must not contain NaN; filter unknown-label rows first."

    if name == "hierarchical":
        return train_hierarchical_model(X_train, y_train, cfg)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.to_numpy(dtype=np.float64))
    classes_sorted = sorted(y_train.unique().tolist())

    quantize_dtype = None
    if name == "full":
        clf = _build_full_classifier(cfg)
        X_fit, y_fit = X_scaled, y_train.to_numpy()
        if bool(cfg.raw["model"]["full"].get("use_smote", True)):
            X_fit, y_fit = _smote_oversample(X_fit, y_fit, cfg.seed)
        clf.fit(X_fit, y_fit)
    else:
        clf = _build_compressed_classifier(cfg)
        max_bal_rows = int(cfg.raw["model"]["compressed"].get("max_balanced_rows", 200_000))
        X_sm, y_sm = _smote_oversample(X_scaled, y_train.to_numpy(), cfg.seed)
        X_bal, y_bal = _balance_by_oversampling(X_sm, y_sm, cfg.seed, max_bal_rows)
        clf.fit(X_bal, y_bal)
        quantize_dtype = cfg.raw["model"]["compressed"].get("quantize_dtype")
        if quantize_dtype:
            clf = _quantize_mlp_weights(clf, quantize_dtype)

    return TrainedModel(
        name=name,
        classifier=clf,
        scaler=scaler,
        classes_=classes_sorted,
        feature_columns=list(X_train.columns),
        quantize_dtype=quantize_dtype,
    )

@dataclass
class HierarchicalClassifier:
    coarse_clf: HistGradientBoostingClassifier
    fine_clfs: Dict[str, HistGradientBoostingClassifier]  
    classes_: List[str] 

    def predict(self, X: np.ndarray) -> np.ndarray:
        coarse_pred = self.coarse_clf.predict(X)
        out = np.empty(len(X), dtype=object)
        for group in ("stationary", "moving"):
            mask = coarse_pred == group
            if not mask.any():
                continue
            out[mask] = self.fine_clfs[group].predict(X[mask])
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        coarse_proba = self.coarse_clf.predict_proba(X)  
        coarse_classes = list(self.coarse_clf.classes_)

        n = len(X)
        out = np.zeros((n, len(self.classes_)), dtype=np.float64)
        class_to_col = {c: i for i, c in enumerate(self.classes_)}

        for group in ("stationary", "moving"):
            if group not in coarse_classes:
                continue
            group_col = coarse_classes.index(group)
            p_group = coarse_proba[:, group_col]  

            fine_clf = self.fine_clfs[group]
            fine_proba = fine_clf.predict_proba(X)  
            for j, cls in enumerate(fine_clf.classes_):
                col = class_to_col[cls]
                out[:, col] = p_group * fine_proba[:, j]

        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        out = out / row_sums
        return out

def train_hierarchical_model(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: Config, verbose: bool = True
) -> TrainedModel:
    def _log(msg: str) -> None:
        if verbose:
            print(f"    [hierarchical] {msg}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.to_numpy(dtype=np.float64))
    y = y_train.to_numpy()

    coarse_y = np.array([_group_for_class(c) for c in y])
    coarse_counts = pd.Series(coarse_y).value_counts().to_dict()
    _log(f"coarse-stage label distribution: {coarse_counts}")
    if len(coarse_counts) < 2:
        _log(
            f"WARNING: only {len(coarse_counts)} coarse group(s) present in this "
            f"training set ({coarse_counts}) -- the coarse classifier cannot learn a "
            f"meaningful stationary-vs-moving split, and every example will be routed "
            f"to a single fine-stage classifier regardless of its true group. This "
            f"would explain badly degraded accuracy; check that y_train actually "
            f"contains both stationary and moving classes."
        )

    coarse_clf = _build_full_classifier_stage(cfg, "coarse")
    coarse_clf.fit(X_scaled, coarse_y)
    coarse_train_acc = float((coarse_clf.predict(X_scaled) == coarse_y).mean())
    _log(f"coarse-stage in-sample accuracy: {coarse_train_acc:.4f}")
    if coarse_train_acc < 0.7:
        _log(
            f"WARNING: coarse-stage in-sample accuracy ({coarse_train_acc:.4f}) is "
            f"low even ON THE TRAINING DATA -- this stage is not learning the "
            f"stationary-vs-moving split well, which will cascade into poor overall "
            f"hierarchical accuracy regardless of how good the fine stages are, since "
            f"every misrouted example is unrecoverable at the fine stage."
        )

    fine_clfs: Dict[str, HistGradientBoostingClassifier] = {}
    use_smote = bool(cfg.raw["model"]["full"].get("use_smote", True))
    for group, group_classes in (("stationary", STATIONARY_GROUP), ("moving", MOVING_GROUP)):
        mask = coarse_y == group
        _log(f"group '{group}': {int(mask.sum())} rows in training set")
        if not mask.any():
            _log(f"group '{group}': SKIPPED (zero rows -- no fine classifier trained for this group)")
            continue
        X_group, y_group = X_scaled[mask], y[mask]

        present_classes = sorted(set(np.unique(y_group).tolist()))
        class_counts_before = pd.Series(y_group).value_counts().to_dict()
        _log(f"group '{group}': class distribution before SMOTE: {class_counts_before}")

        if len(present_classes) < 2:
            _log(
                f"group '{group}': DEGENERATE FALLBACK TRIGGERED -- only "
                f"{len(present_classes)} class present ({present_classes}). Using a "
                f"constant predictor for this entire group instead of a trained "
                f"classifier. If this fires on a real, non-trivial fold, EVERY "
                f"example routed to this group will be predicted as the same single "
                f"class regardless of its features, which can severely degrade "
                f"accuracy for that group's true classes other than the constant one."
            )
            only_class = next(iter(present_classes))
            fine_clfs[group] = _ConstantClassifier(only_class)
            continue

        if use_smote:
            X_group, y_group = _smote_oversample(X_group, y_group, cfg.seed)
            class_counts_after = pd.Series(y_group).value_counts().to_dict()
            _log(f"group '{group}': class distribution after SMOTE: {class_counts_after}")

        stage_name = "fine_stationary" if group == "stationary" else "fine_moving"
        clf = _build_full_classifier_stage(cfg, stage_name)
        clf.fit(X_group, y_group)
        fine_train_acc = float((clf.predict(X_group) == y_group).mean())
        _log(f"group '{group}': fine-stage in-sample accuracy: {fine_train_acc:.4f}")
        if fine_train_acc < 0.5:
            _log(
                f"WARNING: group '{group}' fine-stage in-sample accuracy "
                f"({fine_train_acc:.4f}) is very low even on its own training data."
            )
        fine_clfs[group] = clf

    classes_sorted = sorted(y_train.unique().tolist())
    hier_clf = HierarchicalClassifier(
        coarse_clf=coarse_clf, fine_clfs=fine_clfs, classes_=classes_sorted
    )

    full_pipeline_pred = hier_clf.predict(X_scaled)
    full_pipeline_acc = float((full_pipeline_pred == y).mean())
    _log(f"FULL two-stage pipeline in-sample accuracy: {full_pipeline_acc:.4f}")
    expected_upper_bound = coarse_train_acc  # can't exceed how often routing is correct
    if full_pipeline_acc < expected_upper_bound - 0.15:
        _log(
            f"WARNING: full-pipeline in-sample accuracy ({full_pipeline_acc:.4f}) is "
            f"notably lower than the coarse-stage routing accuracy "
            f"({coarse_train_acc:.4f}) would suggest is achievable -- this points to a "
            f"bug in how predict() combines the coarse and fine stages, rather than "
            f"either stage being individually weak. Compare against each group's "
            f"fine-stage in-sample accuracy reported above."
        )

    return TrainedModel(
        name="hierarchical",
        classifier=hier_clf,
        scaler=scaler,
        classes_=classes_sorted,
        feature_columns=list(X_train.columns),
        quantize_dtype=None,
    )

@dataclass
class _ConstantClassifier:
    only_class: str

    @property
    def classes_(self) -> List[str]:
        return [self.only_class]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.only_class, dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1), dtype=np.float64)

def _balance_by_oversampling(
    X: np.ndarray, y: np.ndarray, seed: int, max_total_rows: int = 200_000
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    target_count = min(int(counts.max()), max(1, max_total_rows // n_classes))
    idx_all = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        if len(idx_c) >= target_count:
            idx_sample = rng.choice(idx_c, size=target_count, replace=False)
        else:
            idx_sample = rng.choice(idx_c, size=target_count, replace=True)
        idx_all.append(idx_sample)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]

def predict(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    X_aligned = X.reindex(columns=model.feature_columns)
    X_aligned = X_aligned.fillna(0.0)
    X_scaled = model.scaler.transform(X_aligned.to_numpy(dtype=np.float64))
    if model.quantize_dtype:
        X_scaled = X_scaled.astype(model.quantize_dtype).astype(np.float64)
    return model.classifier.predict(X_scaled)

def predict_proba(model: TrainedModel, X: pd.DataFrame) -> pd.DataFrame:
    X_aligned = X.reindex(columns=model.feature_columns).fillna(0.0)
    X_scaled = model.scaler.transform(X_aligned.to_numpy(dtype=np.float64))
    proba = model.classifier.predict_proba(X_scaled)
    return pd.DataFrame(proba, index=X.index, columns=model.classifier.classes_)

def save_model(model: TrainedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Path) -> TrainedModel:
    return joblib.load(path)

def model_size_on_disk_bytes(model: TrainedModel) -> int:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.tell()

def model_param_count(model: TrainedModel) -> int:
    clf = model.classifier
    return _classifier_param_count(clf)

def _classifier_param_count(clf: object) -> int:
    total = 0
    if isinstance(clf, HierarchicalClassifier):
        total += _classifier_param_count(clf.coarse_clf)
        for sub_clf in clf.fine_clfs.values():
            total += _classifier_param_count(sub_clf)
        return total
    if hasattr(clf, "coefs_"):  
        for w in clf.coefs_:
            total += w.size
        for b in clf.intercepts_:
            total += b.size
    elif hasattr(clf, "_predictors"):
        try:
            for stage in clf._predictors:
                for tree in stage:
                    total += tree.nodes.shape[0]
        except Exception:
            return -1 
    return total