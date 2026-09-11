from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def categorical_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))

def categorical_macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[List[str]] = None) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

def categorical_balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))

def per_class_prf1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]
) -> Dict[str, Dict[str, float]]:
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {}
    for i, label in enumerate(labels):
        out[label] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
        }
    return out

def compute_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)

@dataclass
class BinaryVerificationMetrics:
    precision: float
    recall: float
    f1: float
    specificity: float
    accuracy: float

def binary_verification_metrics(
    y_true: Sequence[bool], y_pred: Sequence[bool]
) -> BinaryVerificationMetrics:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision == precision and recall == recall and (precision + recall) > 0)
        else float("nan")
    )
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else float("nan")

    return BinaryVerificationMetrics(
        precision=precision, recall=recall, f1=f1, specificity=specificity, accuracy=accuracy
    )

def numeric_accuracy_within_tolerance(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    abs_tolerance: Optional[float] = None,
    rel_tolerance: Optional[float] = None,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")

    abs_err = np.abs(y_pred - y_true)
    correct = np.zeros(len(y_true), dtype=bool)

    if abs_tolerance is not None:
        correct |= abs_err <= abs_tolerance
    if rel_tolerance is not None:
        eps = 1e-9
        rel_err = abs_err / np.maximum(np.abs(y_true), eps)
        correct |= rel_err <= rel_tolerance

    return float(np.mean(correct))

def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred - y_true)))

def mean_absolute_percentage_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    eps = 1e-9
    return float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)))

def interval_iou(pred_start: float, pred_end: float, true_start: float, true_end: float) -> float:
    inter_start = max(pred_start, true_start)
    inter_end = min(pred_end, true_end)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union = max(union_end - union_start, 1e-9)

    return float(intersection / union)

def temporal_precision_recall_f1(
    pred_start: float, pred_end: float, true_start: float, true_end: float
) -> Tuple[float, float, float]:
    inter_start = max(pred_start, true_start)
    inter_end = min(pred_end, true_end)
    intersection = max(0.0, inter_end - inter_start)

    pred_len = max(pred_end - pred_start, 1e-9)
    true_len = max(true_end - true_start, 1e-9)

    precision = intersection / pred_len
    recall = intersection / true_len
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)

def grounding_accuracy_at_threshold(
    ious: Sequence[float], threshold: float
) -> float:
    ious = np.asarray(ious, dtype=np.float64)
    if len(ious) == 0:
        return float("nan")
    return float(np.mean(ious >= threshold))

def accuracy_vs_iou_curve(
    ious: Sequence[float], thresholds: Sequence[float]
) -> Dict[float, float]:
    return {float(t): grounding_accuracy_at_threshold(ious, t) for t in thresholds}

def accuracy_vs_numeric_tolerance_curve(
    y_true: Sequence[float], y_pred: Sequence[float], tolerances: Sequence[float]
) -> Dict[float, float]:
    return {
        float(t): numeric_accuracy_within_tolerance(y_true, y_pred, abs_tolerance=t)
        for t in tolerances
    }

@dataclass
class GroundedAnswerRecord:
    answer_correct: bool
    cited_iou: float
    cited_modality_correct: bool
    cited_channels_correct: bool
    activity_present_in_cited_interval: bool  

def grounded_and_correct(record: GroundedAnswerRecord, iou_threshold: float) -> bool:
    return (
        record.answer_correct
        and record.cited_iou >= iou_threshold
        and record.cited_modality_correct
        and record.cited_channels_correct
    )

def grounding_precision(records: Sequence[GroundedAnswerRecord]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([r.activity_present_in_cited_interval for r in records]))

def evidence_grounding_accuracy(
    records: Sequence[GroundedAnswerRecord], iou_threshold: float
) -> float:
    if not records:
        return float("nan")
    return float(np.mean([grounded_and_correct(r, iou_threshold) for r in records]))

def macro_average_qa_accuracy(per_type_accuracy: Dict[str, float]) -> float:
    vals = [v for v in per_type_accuracy.values() if v == v]  # drop NaNs
    if not vals:
        return float("nan")
    return float(np.mean(vals))

_RUBRIC_SYSTEM_PROMPT = """You are grading the Explanation field of a sensor-grounded
question-answering system's answer to an open-world reasoning question, on a 1-5 scale.

Score 1-5 where:
  1 = explanation does not reference real signal features and/or reasoning is absent or incoherent
  2 = explanation references signal features but they do not support the conclusion
  3 = explanation is plausible but generic; weak connection between signal and conclusion
  4 = explanation clearly cites signal features that support a plausible conclusion
  5 = explanation is precise, cites specific real signal features that strongly and clearly support
      a plausible conclusion

Output STRICT JSON: {"score": <int 1-5>, "reasoning": "<one sentence of grading rationale>"}"""

def score_explanation_with_llm(
    question: str, answer_text: str, explanation_text: str, llm=None
) -> Optional[int]:
    from .llm_client import OllamaClient, OllamaUnavailableError

    llm = llm or OllamaClient.from_config()
    user_prompt = f"Question: {question}\nAnswer: {answer_text}\nExplanation: {explanation_text}"
    try:
        result = llm.chat_json(_RUBRIC_SYSTEM_PROMPT, user_prompt)
        score = int(result.get("score"))
        return max(1, min(5, score))
    except (OllamaUnavailableError, TypeError, ValueError, KeyError):
        return None