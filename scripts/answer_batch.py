from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ask_the_sensors.aggregation import build_timeline  # noqa: E402
from ask_the_sensors.config import load_config  # noqa: E402
from ask_the_sensors.data_io import list_available_uuids, read_user_data  # noqa: E402
from ask_the_sensors.llm_client import OllamaClient  # noqa: E402
from ask_the_sensors.output_format import make_na_answer  # noqa: E402
from ask_the_sensors.preprocessing import prepare_user  # noqa: E402
from ask_the_sensors.qa_engine import answer_question  # noqa: E402
from ask_the_sensors.raw_stream_features import (  # noqa: E402
    RawStreamValidationError,
    build_userdata_from_raw_stream,
)
from ask_the_sensors.recognition_model import load_model, predict, predict_proba  # noqa: E402

def _load_default_model(cfg):
    for model_name in ("hierarchical", "full"):
        model_path = cfg.models_dir / f"fold_0_{model_name}.joblib"
        if model_path.is_file():
            return load_model(model_path)

    raise FileNotFoundError(
        f"No fold_0_hierarchical.joblib or fold_0_full.joblib found in {cfg.models_dir}. "
        f"Run `python -m scripts.train` first, or `python -m scripts.run_all` for the "
        f"full pipeline."
    )

def _build_timeline_for_userdata(ud, model, cfg):
    global_median = pd.Series(0.0, index=model.feature_columns)
    prep = prepare_user(ud, model.feature_columns, global_median, cfg)

    y_pred = predict(model, prep.X)
    proba = predict_proba(model, prep.X)
    confidence = proba.max(axis=1).to_numpy()

    return build_timeline(
        uuid=ud.uuid,
        rel_time_s=prep.rel_time_s,
        predicted_activity=y_pred,
        confidences=confidence,
        gap_before=prep.gap_before,
        X_for_summary=prep.X,
        cfg=cfg,
    )

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path, help="Path to questions.json")
    parser.add_argument("--out", required=True, type=Path, help="Path to write answers.json")
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Optional override: path to a specific trained model .joblib file"
    )
    args = parser.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()

    with open(args.questions, "r") as f:
        questions: List[Dict] = json.load(f)

    model = load_model(args.model_path) if args.model_path else _load_default_model(cfg)

    available_uuids = set(list_available_uuids(cfg))
    llm = OllamaClient.from_config()
    if not llm.is_available():
        print(f"WARNING: Ollama at {llm.cfg.host} (model={llm.cfg.model}) is not reachable. "
              f"Falling back to rule-based intent parsing and templated explanations for "
              f"any question that needs LLM assistance.")

    timeline_cache: Dict[str, object] = {}
    answers = []

    for item in questions:
        qid = item.get("id")
        question = item.get("question")
        uuid = item.get("uuid")
        raw_stream_path = item.get("raw_stream_path")

        if uuid is None and raw_stream_path is None:
            na = make_na_answer(
                f"Question {qid!r} has neither 'uuid' nor 'raw_stream_path'; "
                f"exactly one is required to identify the recording.",
                tier=0,
            )
            result = na.to_dict()
            result = {"id": qid, **{k: v for k, v in result.items() if k != "id"}}
            answers.append(result)
            continue

        if uuid is not None:
            cache_key = ("uuid", uuid)
            if uuid not in available_uuids:
                na = make_na_answer(
                    f"Recording for UUID {uuid} not found under {cfg.primary_data_dir}.",
                    tier=0,
                )
                result = na.to_dict()
                result = {"id": qid, **{k: v for k, v in result.items() if k != "id"}}
                answers.append(result)
                continue
            if cache_key not in timeline_cache:
                ud = read_user_data(uuid, cfg)
                timeline_cache[cache_key] = _build_timeline_for_userdata(ud, model, cfg)
        else:
            cache_key = ("raw_stream", raw_stream_path)
            if cache_key not in timeline_cache:
                recording_id = item.get("recording_id") or Path(raw_stream_path).stem
                try:
                    ud = build_userdata_from_raw_stream(Path(raw_stream_path), uuid=recording_id)
                except RawStreamValidationError as exc:
                    na = make_na_answer(str(exc), tier=0)
                    result = na.to_dict()
                    result = {"id": qid, **{k: v for k, v in result.items() if k != "id"}}
                    answers.append(result)
                    continue
                timeline_cache[cache_key] = _build_timeline_for_userdata(ud, model, cfg)

        timeline = timeline_cache[cache_key]

        structured = answer_question(question, timeline, cfg=cfg, llm=llm)
        result = structured.to_dict()
        result["id"] = qid
        # Reorder id to the front for readability.
        result = {"id": qid, **{k: v for k, v in result.items() if k != "id"}}
        answers.append(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(answers, f, indent=2)

    print(f"Wrote {len(answers)} answers to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())