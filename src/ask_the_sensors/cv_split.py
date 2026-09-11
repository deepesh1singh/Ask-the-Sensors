from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from .config import Config, load_config

@dataclass
class FoldSplit:
    fold_index: int
    train_uuids: List[str]
    test_uuids: List[str]

def _read_uuid_list(path: Path) -> List[str]:
    if not path.is_file():
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_cv_folds(cfg: Optional[Config] = None, n_folds: Optional[int] = None) -> List[FoldSplit]:
    cfg = cfg or load_config()
    n_folds = n_folds or int(cfg.raw["cross_validation"]["n_folds"])
    fold_dir = cfg.cv_folds_dir

    if not fold_dir.is_dir():
        raise FileNotFoundError(
            f"CV folds directory not found: {fold_dir}\n"
            f"Download cv5Folds.zip from http://extrasensory.ucsd.edu/ and run "
            f"`python -m scripts.setup_data` (see README.md section 2)."
        )

    folds: List[FoldSplit] = []
    for i in range(n_folds):
        train = (
            _read_uuid_list(fold_dir / f"fold_{i}_train_iphone_uuids.txt")
            + _read_uuid_list(fold_dir / f"fold_{i}_train_android_uuids.txt")
        )
        test = (
            _read_uuid_list(fold_dir / f"fold_{i}_test_iphone_uuids.txt")
            + _read_uuid_list(fold_dir / f"fold_{i}_test_android_uuids.txt")
        )
        if not train and not test:
            raise FileNotFoundError(
                f"No UUID files found for fold {i} in {fold_dir}. "
                f"Expected files like 'fold_{i}_train_iphone_uuids.txt'. "
                f"Check that cv5Folds.zip was unzipped directly into this directory."
            )

        train_set, test_set = set(train), set(test)
        overlap = train_set & test_set
        if overlap:
            raise ValueError(
                f"Fold {i}: {len(overlap)} UUID(s) appear in both train and test "
                f"sets, violating the user-level split requirement: {sorted(overlap)[:5]}..."
            )

        folds.append(FoldSplit(fold_index=i, train_uuids=sorted(train_set), test_uuids=sorted(test_set)))

    return folds

def restrict_folds_to_available(
    folds: List[FoldSplit], available_uuids: Set[str]
) -> List[FoldSplit]:
    restricted = []
    for fold in folds:
        train = [u for u in fold.train_uuids if u in available_uuids]
        test = [u for u in fold.test_uuids if u in available_uuids]
        restricted.append(FoldSplit(fold_index=fold.fold_index, train_uuids=train, test_uuids=test))
    return restricted

def summarize_folds(folds: List[FoldSplit]) -> str:
    lines = []
    for fold in folds:
        lines.append(
            f"Fold {fold.fold_index}: train={len(fold.train_uuids)} users, "
            f"test={len(fold.test_uuids)} users"
        )
    return "\n".join(lines)

if __name__ == "__main__":
    cfg = load_config()
    folds = load_cv_folds(cfg)
    print(summarize_folds(folds))