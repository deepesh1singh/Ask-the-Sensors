from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .aggregation import (
    Interval,
    Timeline,
    activity_at_time,
    activity_totals,
    all_intervals_for_activity,
    compare_durations,
    count_occurrences,
    first_onset,
    is_prolonged,
    longest_interval,
    total_duration_for_activity,
)
from .config import Config, load_config
from .evidence import EvidenceBlock, build_evidence, build_na_evidence, describe_signal, format_timestamp_ranges
from .llm_client import OllamaClient, OllamaUnavailableError
from .output_format import StructuredAnswer, make_answer, make_na_answer

_ACTIVITY_ALIASES: Dict[str, List[str]] = {
    "LYING_DOWN": ["lying down", "lie down", "lying", "resting", "asleep", "sleeping"],
    "SITTING": ["sitting", "sit down", "seated"],
    "OR_standing": ["standing in place", "standing still", "standing"],
    "STANDING_AND_MOVING": ["standing and moving", "standing while moving", "moving around while standing"],
    "FIX_walking": ["walking", "walk", "her walk", "his walk"],
    "FIX_running": ["running", "run", "jogging"],
    "BICYCLING": ["bicycling", "biking", "cycling", "riding a bike", "pedal", "wheeled"],
}

_STRENUOUS_ACTIVITIES = {"FIX_running", "BICYCLING", "STANDING_AND_MOVING", "FIX_walking"}
_PROLONGED_THRESHOLD_S = 1200  

def _find_activities_in_text(text: str) -> List[str]:
    text_l = text.lower()
    found = []
    for canon, aliases in _ACTIVITY_ALIASES.items():
        for alias in aliases:
            if alias in text_l:
                found.append(canon)
                break
    return found

def _display_name(cfg: Config, canon: str) -> str:
    return cfg.activity_display_names.get(canon, canon)

@dataclass
class Intent:
    tier: int
    operation: str  
    activities: List[str]
    raw_question: str
    time_query_s: Optional[int] = None

_DURATION_PATTERNS = [
    r"how long (was|did|has) the user (?:been )?(\w[\w\s]*)",
    r"how much time",
    r"how many minutes",
    r"duration of",
]
_COUNT_PATTERNS = [
    r"how many times",
    r"how often",
    r"number of times",
]
_COMPARE_PATTERNS = [
    r"more time (\w[\w\s]*) or (\w[\w\s]*)",
    r"(\w[\w\s]*) or (\w[\w\s]*)\??$",
    r"compare",
]
_ONSET_PATTERNS = [
    r"begin",
    r"start(ed)?",
    r"when did",
    r"onset",
]
_VERIFY_PATTERNS = [
    r"^is the user",
    r"^was the user",
    r"^did the user",
    r"\bany point\b",
]
_PROLONGED_PATTERNS = [
    r"prolonged",
    r"for a (long|extended) (period|time)",
]
_PLAUSIBILITY_PATTERNS = [
    r"consistent with",
    r"wheeled",
    r"pedal",
    r"strenuous",
    r"unsteady",
    r"doing anything",
]

def parse_intent_rule_based(question: str) -> Optional[Intent]:
    q = question.strip()
    q_l = q.lower()
    activities = _find_activities_in_text(q)

    if any(re.search(p, q_l) for p in _PROLONGED_PATTERNS):
        return Intent(tier=4, operation="prolonged", activities=activities, raw_question=q)

    if any(re.search(p, q_l) for p in _PLAUSIBILITY_PATTERNS):
        return Intent(tier=4, operation="plausibility", activities=activities, raw_question=q)

    if any(re.search(p, q_l) for p in _ONSET_PATTERNS) and activities:
        return Intent(tier=3, operation="onset", activities=activities, raw_question=q)

    if any(re.search(p, q_l) for p in _COMPARE_PATTERNS) and len(activities) >= 2:
        return Intent(tier=2, operation="compare", activities=activities[:2], raw_question=q)

    if any(re.search(p, q_l) for p in _COUNT_PATTERNS):
        return Intent(tier=2, operation="count", activities=activities, raw_question=q)

    if any(re.search(p, q_l) for p in _DURATION_PATTERNS):
        return Intent(tier=2, operation="duration", activities=activities, raw_question=q)

    if any(re.search(p, q_l) for p in _VERIFY_PATTERNS) and activities:
        return Intent(tier=1, operation="verify", activities=activities, raw_question=q)

    if "what activity" in q_l or "what is the user doing" in q_l or "what was the user doing" in q_l:
        return Intent(tier=1, operation="identify", activities=[], raw_question=q)

    if activities and ("doing" in q_l or "?" in q_l):
        return Intent(tier=1, operation="verify", activities=activities, raw_question=q)

    return None

_INTENT_SYSTEM_PROMPT = """You are an intent parser for a sensor question-answering system.
Given a user's natural-language question about a wearable sensor recording,
output STRICT JSON with this schema:

{
  "tier": <1|2|3|4>,
  "operation": one of ["identify","verify","duration","count","compare","onset","prolonged","plausibility"],
  "activities": [list of zero or more of: "LYING_DOWN","SITTING","OR_standing","STANDING_AND_MOVING","FIX_walking","FIX_running","BICYCLING"]
}

Guidance:
- "identify": asks what activity the user is doing overall (no specific activity named).
- "verify": asks yes/no whether the user is/was doing a specific named activity.
- "duration": asks how long / how much time an activity took.
- "count": asks how many times / how often something happened.
- "compare": asks which of two activities took more/less time.
- "onset": asks when something began/started.
- "prolonged": asks whether an activity happened for a long/extended period.
- "plausibility": open-world reasoning about behavior not necessarily one of the 7 fixed labels
  (e.g. wheeled/pedal-based movement, strenuous activity, consistency with a scenario).

Only output the JSON object, nothing else."""

def parse_intent(question: str, llm: Optional[OllamaClient] = None) -> Intent:
    rule_based = parse_intent_rule_based(question)
    if rule_based is not None:
        return rule_based

    if llm is None:
        llm = OllamaClient.from_config()

    try:
        result = llm.chat_json(_INTENT_SYSTEM_PROMPT, question)
        tier = int(result.get("tier", 4))
        operation = str(result.get("operation", "plausibility"))
        activities = [a for a in result.get("activities", []) if a in _ACTIVITY_ALIASES]
        return Intent(tier=tier, operation=operation, activities=activities, raw_question=question)
    except OllamaUnavailableError:
        return Intent(
            tier=4,
            operation="plausibility",
            activities=_find_activities_in_text(question),
            raw_question=question,
        )

_EXPLANATION_SYSTEM_PROMPT = """You are writing the Explanation field for a sensor-grounded
question-answering system. You will be given the question, the computed answer, the
activity name(s), and a deterministic, pre-computed description of the accelerometer/
gyroscope signal pattern for the cited time interval(s). Write ONE to TWO sentences of
plain-English reasoning that ties the cited signal pattern to the conclusion.

Rules:
- Do NOT invent any numbers, timestamps, or signal characteristics that are not present
  in the provided signal description.
- Do NOT mention channel-by-channel raw feature names; paraphrase the described pattern
  in natural, clinical-report style language (as if explaining to a physiotherapist).
- Keep it concise and concrete.

Output STRICT JSON: {"explanation": "<your text>"}"""

def phrase_explanation(
    question: str,
    answer_text: str,
    activity_display: str,
    signal_descriptions: List[str],
    llm: Optional[OllamaClient] = None,
) -> str:
    if not signal_descriptions:
        return "N/A"

    joined_desc = " | ".join(signal_descriptions[:3])
    user_prompt = (
        f"Question: {question}\n"
        f"Answer: {answer_text}\n"
        f"Activity: {activity_display}\n"
        f"Signal description(s) for cited interval(s): {joined_desc}"
    )

    if llm is None:
        llm = OllamaClient.from_config()

    try:
        result = llm.chat_json(_EXPLANATION_SYSTEM_PROMPT, user_prompt)
        text = str(result.get("explanation", "")).strip()
        if text:
            return text
    except OllamaUnavailableError:
        pass

    return (
        f"Based on the recognized {activity_display.lower()} interval(s), the signal shows "
        f"{joined_desc}, which is consistent with the stated answer."
    )

def answer_identify(timeline: Timeline, cfg: Config) -> StructuredAnswer:
    if not timeline.intervals:
        return make_na_answer("No recognized activity intervals in this recording.", tier=1)
    iv = timeline.intervals[-1]
    display = _display_name(cfg, iv.activity)
    return make_answer(
        answer=display,
        activity_event=display,
        evidence=None,
        explanation="N/A",
        tier=1,
    )

def answer_verify(timeline: Timeline, cfg: Config, activity: str) -> StructuredAnswer:
    _, matched = count_occurrences(timeline, activity)
    display = _display_name(cfg, activity)
    yes = len(matched) > 0
    return make_answer(
        answer="Yes" if yes else "No",
        activity_event=display,
        evidence=None,
        explanation="N/A",
        tier=1,
    )

def answer_duration(
    timeline: Timeline, cfg: Config, activity: str, question: str, llm: Optional[OllamaClient]
) -> StructuredAnswer:
    total, matched = total_duration_for_activity(timeline, activity)
    display = _display_name(cfg, activity)

    if not matched:
        return make_answer(
            answer="0 seconds",
            activity_event=display,
            evidence=build_na_evidence(),
            explanation=f"No {display.lower()} intervals were detected in this recording.",
            tier=2,
        )

    evidence = build_evidence(matched, modality="both", channels="All")
    interval_desc = ", ".join(f"{iv.duration_s}s" for iv in matched)
    explanation_seed = (
        f"{display} was detected in {len(matched)} interval(s) "
        f"({interval_desc}), summing to {total} seconds."
    )
    return make_answer(
        answer=f"{total} seconds",
        activity_event=display,
        evidence=evidence,
        explanation=explanation_seed,
        tier=2,
    )

def answer_count(timeline: Timeline, cfg: Config, activity: str) -> StructuredAnswer:
    n, matched = count_occurrences(timeline, activity)
    display = _display_name(cfg, activity)
    evidence = build_evidence(matched, modality="both", channels="All") if matched else build_na_evidence()
    explanation = (
        f"{display} occurred as {n} separate interval(s) in the recognized activity timeline."
        if n > 0
        else f"No {display.lower()} intervals were detected."
    )
    return make_answer(
        answer=str(n),
        activity_event=display,
        evidence=evidence,
        explanation=explanation,
        tier=2,
    )

def answer_compare(
    timeline: Timeline, cfg: Config, activity_a: str, activity_b: str
) -> StructuredAnswer:
    winner, dur_a, dur_b, ivs_a, ivs_b = compare_durations(timeline, activity_a, activity_b)
    display_a, display_b = _display_name(cfg, activity_a), _display_name(cfg, activity_b)

    if winner is None:
        answer_text = f"Equal ({display_a} and {display_b} both totalled {dur_a} seconds)"
        activity_event = f"{display_a}, {display_b}"
    else:
        winner_display = display_a if winner == activity_a else display_b
        answer_text = winner_display
        activity_event = f"{display_a}, {display_b}"

    all_ivs = ivs_a + ivs_b
    if all_ivs:
        evidence = EvidenceBlock(
            timestamps=f"{display_a} = {dur_a} seconds total, {display_b} = {dur_b} seconds total",
            sensor_modality="Accelerometer, Gyroscope",
            sensor_channels="All",
        )
    else:
        evidence = build_na_evidence()

    explanation = (
        f"Total {display_a.lower()} time ({dur_a}s) "
        f"{'exceeded' if dur_a > dur_b else ('was less than' if dur_a < dur_b else 'equalled')} "
        f"total {display_b.lower()} time ({dur_b}s) over the recording."
    )
    return make_answer(
        answer=answer_text,
        activity_event=activity_event,
        evidence=evidence,
        explanation=explanation,
        tier=2,
    )

def answer_onset(
    timeline: Timeline, cfg: Config, activity: str, question: str, llm: Optional[OllamaClient]
) -> StructuredAnswer:
    iv = first_onset(timeline, activity)
    display = _display_name(cfg, activity)

    if iv is None:
        return make_answer(
            answer="No",
            activity_event=f"Onset of {display.lower()}",
            evidence=build_na_evidence(),
            explanation=f"No {display.lower()} interval was detected in this recording.",
            tier=3,
        )

    evidence = build_evidence([iv], modality="both", channels="All")
    signal_desc = describe_signal(iv)
    answer_text = f"Yes, {display.lower()} began at {iv.start_s} seconds"
    explanation = phrase_explanation(
        question=question,
        answer_text=answer_text,
        activity_display=display,
        signal_descriptions=[signal_desc],
        llm=llm,
    )
    return make_answer(
        answer=answer_text,
        activity_event=f"Onset of {display.lower()}",
        evidence=evidence,
        explanation=explanation,
        tier=3,
    )

def answer_prolonged(
    timeline: Timeline, cfg: Config, activity: str, question: str, llm: Optional[OllamaClient]
) -> StructuredAnswer:
    display = _display_name(cfg, activity) if activity else "the activity in question"
    iv = longest_interval(timeline, activity if activity else None)

    if iv is None:
        return make_answer(
            answer="No",
            activity_event=f"Prolonged {display.lower()}",
            evidence=build_na_evidence(),
            explanation=f"No relevant intervals were detected in this recording.",
            tier=4,
        )

    prolonged = is_prolonged(iv, _PROLONGED_THRESHOLD_S)
    evidence = build_evidence([iv], modality="both", channels="All")
    signal_desc = describe_signal(iv)
    answer_text = "Likely yes" if prolonged else "Likely no"
    explanation = phrase_explanation(
        question=question,
        answer_text=answer_text,
        activity_display=display,
        signal_descriptions=[
            f"{signal_desc}; interval duration {iv.duration_s} seconds "
            f"({'at or above' if prolonged else 'below'} the {_PROLONGED_THRESHOLD_S}s "
            f"prolonged-activity threshold used by this system)"
        ],
        llm=llm,
    )
    return make_answer(
        answer=answer_text,
        activity_event=f"Prolonged {display.lower()}" if activity else "Prolonged activity",
        evidence=evidence,
        explanation=explanation,
        tier=4,
    )

_WHEELED_ALIASES = {"BICYCLING"}
_STRENUOUS_QUESTION_HINTS = ["strenuous", "unsteady", "vigorous", "intense"]

def answer_plausibility(
    timeline: Timeline, cfg: Config, activities: List[str], question: str, llm: Optional[OllamaClient]
) -> StructuredAnswer:
    q_l = question.lower()

    if "wheel" in q_l or "pedal" in q_l or activities and set(activities) & _WHEELED_ALIASES:
        target = "BICYCLING"
        matched = all_intervals_for_activity(timeline, target)
        display = _display_name(cfg, target)
        if not matched:
            return make_answer(
                answer="No",
                activity_event="Wheeled or pedal-based movement",
                evidence=build_na_evidence(),
                explanation="No signal pattern consistent with cycling was detected in this recording.",
                tier=4,
            )
        iv = longest_interval(timeline, target)
        evidence = build_evidence([iv], modality="both", channels="All")
        signal_desc = describe_signal(iv)
        answer_text = "Yes"
        explanation = phrase_explanation(
            question=question,
            answer_text=answer_text,
            activity_display="Unknown outdoor physical activity, consistent with cycling",
            signal_descriptions=[signal_desc],
            llm=llm,
        )
        return make_answer(
            answer=answer_text,
            activity_event="Unknown outdoor physical activity, consistent with cycling",
            evidence=evidence,
            explanation=explanation,
            tier=4,
        )

    if any(h in q_l for h in _STRENUOUS_QUESTION_HINTS):
        strenuous_intervals = [
            iv for iv in timeline.intervals if iv.activity in _STRENUOUS_ACTIVITIES
        ]
        if not strenuous_intervals:
            return make_answer(
                answer="No",
                activity_event="Strenuous activity",
                evidence=build_na_evidence(),
                explanation="No intervals of running, bicycling, walking, or standing-and-moving were detected.",
                tier=4,
            )
        evidence = build_evidence(strenuous_intervals, modality="both", channels="All")
        descs = [describe_signal(iv) for iv in strenuous_intervals[:3]]
        answer_text = "Yes"
        explanation = phrase_explanation(
            question=question,
            answer_text=answer_text,
            activity_display="Strenuous activity",
            signal_descriptions=descs,
            llm=llm,
        )
        return make_answer(
            answer=answer_text,
            activity_event="Strenuous activity",
            evidence=evidence,
            explanation=explanation,
            tier=4,
        )

    if activities:
        matched_all: List[Interval] = []
        for a in activities:
            matched_all.extend(all_intervals_for_activity(timeline, a))
        if not matched_all:
            return make_answer(
                answer="No",
                activity_event=", ".join(_display_name(cfg, a) for a in activities),
                evidence=build_na_evidence(),
                explanation="No matching intervals were detected in this recording.",
                tier=4,
            )
        evidence = build_evidence(matched_all, modality="both", channels="All")
        descs = [describe_signal(iv) for iv in matched_all[:3]]
        explanation = phrase_explanation(
            question=question,
            answer_text="Yes",
            activity_display=", ".join(_display_name(cfg, a) for a in activities),
            signal_descriptions=descs,
            llm=llm,
        )
        return make_answer(
            answer="Yes",
            activity_event=", ".join(_display_name(cfg, a) for a in activities),
            evidence=evidence,
            explanation=explanation,
            tier=4,
        )

    return make_na_answer(
        "This open-world question could not be mapped to a supported operation "
        "or any of the 7 known activity classes.",
        tier=4,
    )

def answer_question(
    question: str,
    timeline: Timeline,
    cfg: Optional[Config] = None,
    llm: Optional[OllamaClient] = None,
) -> StructuredAnswer:
    cfg = cfg or load_config()
    llm = llm or OllamaClient.from_config()

    intent = parse_intent(question, llm=llm)

    if intent.operation == "identify":
        return answer_identify(timeline, cfg)

    if intent.operation == "verify":
        if not intent.activities:
            return make_na_answer("Could not identify which activity the question refers to.", tier=1)
        return answer_verify(timeline, cfg, intent.activities[0])

    if intent.operation == "duration":
        if not intent.activities:
            return make_na_answer("Could not identify which activity the question refers to.", tier=2)
        return answer_duration(timeline, cfg, intent.activities[0], question, llm)

    if intent.operation == "count":
        if not intent.activities:
            return make_na_answer("Could not identify which activity the question refers to.", tier=2)
        return answer_count(timeline, cfg, intent.activities[0])

    if intent.operation == "compare":
        if len(intent.activities) < 2:
            return make_na_answer("Could not identify two activities to compare.", tier=2)
        return answer_compare(timeline, cfg, intent.activities[0], intent.activities[1])

    if intent.operation == "onset":
        if not intent.activities:
            return make_na_answer("Could not identify which activity the question refers to.", tier=3)
        return answer_onset(timeline, cfg, intent.activities[0], question, llm)

    if intent.operation == "prolonged":
        activity = intent.activities[0] if intent.activities else None
        return answer_prolonged(timeline, cfg, activity, question, llm)

    if intent.operation == "plausibility":
        return answer_plausibility(timeline, cfg, intent.activities, question, llm)

    return make_na_answer(f"Unrecognized question operation: {intent.operation!r}.", tier=intent.tier)