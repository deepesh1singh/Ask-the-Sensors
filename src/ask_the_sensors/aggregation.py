from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .config import Config, load_config

@dataclass
class Interval:
    activity: str
    start_s: int            
    end_s: int                
    n_windows: int
    mean_confidence: float
    signal_summary: Dict[str, float] = field(default_factory=dict)
    @property
    def duration_s(self) -> int:
        return int(self.end_s - self.start_s)

@dataclass
class Timeline:
    uuid: str
    intervals: List[Interval]
    total_duration_s: int
    example_duration_s: int

def build_intervals(
    rel_time_s: np.ndarray,
    predicted_activity: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    gap_before: Optional[np.ndarray] = None,
    X_for_summary: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
) -> List[Interval]:
    cfg = cfg or load_config()
    example_duration_s = int(cfg.example_duration_seconds)

    n = len(rel_time_s)
    if n == 0:
        return []

    if confidences is None:
        confidences = np.ones(n, dtype=np.float64)
    if gap_before is None:
        gap_before = np.zeros(n, dtype=bool)
        gap_before[0] = True

    intervals: List[Interval] = []
    run_start_idx = 0

    def _flush(start_idx: int, end_idx_inclusive: int) -> None:
        act = predicted_activity[start_idx]
        start_s = int(rel_time_s[start_idx])
        end_s = int(rel_time_s[end_idx_inclusive]) + example_duration_s
        n_windows = end_idx_inclusive - start_idx + 1
        mean_conf = float(np.mean(confidences[start_idx : end_idx_inclusive + 1]))

        summary: Dict[str, float] = {}
        if X_for_summary is not None:
            block = X_for_summary.iloc[start_idx : end_idx_inclusive + 1]
            for col in block.columns:
                summary[col] = float(np.nanmean(block[col].to_numpy()))

        intervals.append(
            Interval(
                activity=act,
                start_s=start_s,
                end_s=end_s,
                n_windows=n_windows,
                mean_confidence=mean_conf,
                signal_summary=summary,
            )
        )

    for i in range(1, n):
        same_activity = predicted_activity[i] == predicted_activity[i - 1]
        contiguous = not gap_before[i]
        if same_activity and contiguous:
            continue
        _flush(run_start_idx, i - 1)
        run_start_idx = i
    _flush(run_start_idx, n - 1)
    return intervals

def build_timeline(
    uuid: str,
    rel_time_s: np.ndarray,
    predicted_activity: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    gap_before: Optional[np.ndarray] = None,
    X_for_summary: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
) -> Timeline:
    cfg = cfg or load_config()
    intervals = build_intervals(
        rel_time_s, predicted_activity, confidences, gap_before, X_for_summary, cfg
    )
    total_duration = int(rel_time_s.max() - rel_time_s.min()) + cfg.example_duration_seconds if len(rel_time_s) else 0
    return Timeline(
        uuid=uuid,
        intervals=intervals,
        total_duration_s=total_duration,
        example_duration_s=cfg.example_duration_seconds,
    )

def total_duration_for_activity(timeline: Timeline, activity: str) -> Tuple[int, List[Interval]]:
    matched = [iv for iv in timeline.intervals if iv.activity == activity]
    total = sum(iv.duration_s for iv in matched)
    return total, matched

def count_occurrences(timeline: Timeline, activity: str) -> Tuple[int, List[Interval]]:
    matched = [iv for iv in timeline.intervals if iv.activity == activity]
    return len(matched), matched

def compare_durations(timeline: Timeline, activity_a: str, activity_b: str):
    dur_a, ivs_a = total_duration_for_activity(timeline, activity_a)
    dur_b, ivs_b = total_duration_for_activity(timeline, activity_b)
    if dur_a > dur_b:
        winner = activity_a
    elif dur_b > dur_a:
        winner = activity_b
    else:
        winner = None  
    return winner, dur_a, dur_b, ivs_a, ivs_b

def first_onset(timeline: Timeline, activity: str) -> Optional[Interval]:
    for iv in timeline.intervals:
        if iv.activity == activity:
            return iv
    return None

def all_intervals_for_activity(timeline: Timeline, activity: str) -> List[Interval]:
    return [iv for iv in timeline.intervals if iv.activity == activity]

def activity_at_time(timeline: Timeline, t_s: int) -> Optional[Interval]:
    for iv in timeline.intervals:
        if iv.start_s <= t_s < iv.end_s:
            return iv
    return None

def activity_in_window(timeline: Timeline, start_s: int, end_s: int) -> List[Interval]:
    out = []
    for iv in timeline.intervals:
        if iv.start_s < end_s and iv.end_s > start_s:
            out.append(iv)
    return out

def is_prolonged(interval: Interval, threshold_s: int = 1200) -> bool:
    return interval.duration_s >= threshold_s

def longest_interval(timeline: Timeline, activity: Optional[str] = None) -> Optional[Interval]:
    candidates = timeline.intervals if activity is None else all_intervals_for_activity(timeline, activity)
    if not candidates:
        return None
    return max(candidates, key=lambda iv: iv.duration_s)

def activity_totals(timeline: Timeline, activity_classes: List[str]) -> Dict[str, int]:
    totals = {a: 0 for a in activity_classes}
    for iv in timeline.intervals:
        if iv.activity in totals:
            totals[iv.activity] += iv.duration_s
    return totals