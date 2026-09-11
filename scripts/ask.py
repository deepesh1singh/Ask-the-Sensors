"""Interactive single-question CLI.

Usage (ExtraSensory-corpus recording, by UUID):
    python -m scripts.ask --uuid <UUID> --question "How long was the user walking?"

Usage (raw accelerometer+gyroscope stream -- the brief's own stated
system input format; see raw_stream_features.py for the exact CSV schema
expected and the full rationale for why this mode exists):
    python -m scripts.ask --raw-stream path/to/recording.csv \
        --question "How long was the user walking?" [--recording-id my-label]

Exactly one of --uuid / --raw-stream must be given.

Uses fold 0's trained 'full' model to recognize the activity timeline for
the given user's whole recording, then routes the question through the
interface layer (qa_engine.py) to produce the mandated structured answer.

If no trained model is found, this script trains one on the fly (fold 0
convention: fits on fold 0's training users, applies to the requested
user's own recording) so the script is always runnable standalone.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from ask_the_sensors.aggregation import build_timeline  
from ask_the_sensors.config import load_config 
from ask_the_sensors.cv_split import load_cv_folds, restrict_folds_to_available
from ask_the_sensors.data_io import list_available_uuids, read_all_users, read_user_data  
from ask_the_sensors.llm_client import OllamaClient  
from ask_the_sensors.preprocessing import prepare_user, select_stable_feature_columns 
from ask_the_sensors.qa_engine import answer_question 
from ask_the_sensors.raw_stream_features import (  
    RawStreamValidationError,
    build_userdata_from_raw_stream,
)
from ask_the_sensors.recognition_model import load_model, predict, predict_proba, train_model  

def _get_or_train_fold0_model(cfg):
    for model_name in ("hierarchical", "full"):
        model_path = cfg.models_dir / f"fold_0_{model_name}.joblib"
        if model_path.is_file():
            print(f"Loading trained model from {model_path}")
            return load_model(model_path)

    print("No trained model found; training a fold-0 hierarchical model on the fly "
          "(run `python -m scripts.train` beforehand to avoid this delay).")
    uuids = list_available_uuids(cfg)
    folds = restrict_folds_to_available(load_cv_folds(cfg), set(uuids))
    fold0 = next(f for f in folds if f.fold_index == 0)

    users = read_all_users(fold0.train_uuids, cfg)
    feature_columns = select_stable_feature_columns(users, cfg)

    prefixes = tuple(cfg.feature_prefixes.values())
    all_feats = pd.concat(
        [
            users[u].features[[c for c in users[u].features.columns if c.split(":")[0] in prefixes]].reindex(
                columns=feature_columns
            )
            for u in users
        ],
        axis=0,
    )
    global_median = all_feats.median(axis=0, skipna=True).fillna(0.0)

    X_list, y_list = [], []
    for u, ud in users.items():
        prep = prepare_user(ud, feature_columns, global_median, cfg)
        mask = prep.y.notna()
        if mask.sum() == 0:
            continue
        X_list.append(prep.X.loc[mask])
        y_list.append(prep.y.loc[mask])
    X_train = pd.concat(X_list, axis=0)
    y_train = pd.concat(y_list, axis=0)

    model = train_model("hierarchical", X_train, y_train, cfg)
    from ask_the_sensors.recognition_model import save_model
    save_path = cfg.models_dir / "fold_0_hierarchical.joblib"
    save_model(model, save_path)
    print(f"Trained and saved model to {save_path}")
    return model

def main() -> int:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--uuid", help="ExtraSensory user UUID (reads <UUID>.features_labels.csv[.gz] "
                                             f"from the primary data directory)")
    input_group.add_argument("--raw-stream", type=Path,
                              help="Path to a raw accelerometer+gyroscope stream CSV (columns: "
                                   "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z). This is the "
                                   "brief's own stated system input format -- see "
                                   "raw_stream_features.py for the exact schema.")
    parser.add_argument("--recording-id", default="raw-stream-recording",
                         help="Display label for a --raw-stream recording (used in place of a UUID "
                              "in the printed output). Ignored when --uuid is given.")
    parser.add_argument("--question", required=True, help="Natural-language question")
    args = parser.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()

    model = _get_or_train_fold0_model(cfg)

    if args.uuid is not None:
        available = list_available_uuids(cfg)
        if args.uuid not in available:
            print(f"ERROR: UUID {args.uuid} not found among {len(available)} available users in "
                  f"{cfg.primary_data_dir}.")
            if available:
                print(f"Example available UUIDs: {available[:3]}")
            return 1
        ud = read_user_data(args.uuid, cfg)
        recording_id = args.uuid
    else:
        try:
            ud = build_userdata_from_raw_stream(args.raw_stream, uuid=args.recording_id)
        except RawStreamValidationError as exc:
            print(f"ERROR: {exc}")
            return 1
        recording_id = args.recording_id

    prep = prepare_user(ud, model.feature_columns, pd.Series(0.0, index=model.feature_columns), cfg)

    y_pred = predict(model, prep.X)
    proba = predict_proba(model, prep.X)
    confidence = proba.max(axis=1).to_numpy()

    timeline = build_timeline(
        uuid=recording_id,
        rel_time_s=prep.rel_time_s,
        predicted_activity=y_pred,
        confidences=confidence,
        gap_before=prep.gap_before,
        X_for_summary=prep.X,
        cfg=cfg,
    )

    print(f"\nRecognized {len(timeline.intervals)} activity intervals over "
          f"{timeline.total_duration_s} seconds of recording for user {recording_id}.\n")

    llm = OllamaClient.from_config()
    if not llm.is_available():
        print(f"NOTE: Ollama server at {llm.cfg.host} is not reachable; "
              f"falling back to rule-based intent parsing and templated explanations.\n")

    structured = answer_question(args.question, timeline, cfg=cfg, llm=llm)

    print("=" * 60)
    print(structured.to_text())
    print("=" * 60)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())