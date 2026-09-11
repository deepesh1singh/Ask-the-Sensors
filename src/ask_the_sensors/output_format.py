from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .evidence import EvidenceBlock, build_na_evidence

TIMESTAMP_CONVENTION = "seconds from the start of the recording"

@dataclass
class StructuredAnswer:
    answer: str
    activity_event: str
    evidence: EvidenceBlock
    explanation: str
    tier: int  
    def to_text(self) -> str:
        ev = self.evidence
        lines = [
            f"Answer: {self.answer}",
            f"Activity/Event: {self.activity_event}",
            "Evidence:",
            f" Timestamp(s): {ev.timestamps}",
            f" Sensor Modality: {ev.sensor_modality}",
            f" Sensor Channel(s): {ev.sensor_channels}",
            f"Explanation: {self.explanation}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "Answer": self.answer,
            "Activity/Event": self.activity_event,
            "Evidence": self.evidence.to_dict(),
            "Explanation": self.explanation,
            "_tier": self.tier,
            "_timestamp_convention": TIMESTAMP_CONVENTION,
        }

def make_answer(
    answer: str,
    activity_event: str,
    evidence: Optional[EvidenceBlock],
    explanation: str,
    tier: int,
) -> StructuredAnswer:
    return StructuredAnswer(
        answer=answer,
        activity_event=activity_event,
        evidence=evidence or build_na_evidence(),
        explanation=explanation,
        tier=tier,
    )

def make_na_answer(reason: str, tier: int) -> StructuredAnswer:
    return StructuredAnswer(
        answer="N/A",
        activity_event="N/A",
        evidence=build_na_evidence(),
        explanation=reason,
        tier=tier,
    )