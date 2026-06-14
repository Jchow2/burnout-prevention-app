"""
src/scoring/burnout_scales.py

Scale helpers for BAT-12 and OLBI.
- Scores each scale independently.
- Normalizes subscale means to 0-10.
- Keeps raw means for psychometric auditability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_mean_to_0_10(mean: Optional[float], min_val: int, max_val: int) -> Optional[float]:
    if mean is None:
        return None
    if max_val == min_val:
        raise ValueError("max_val must differ from min_val")
    return clamp(((mean - min_val) / (max_val - min_val)) * 10.0, 0.0, 10.0)


def reverse_score(value: int, min_val: int, max_val: int) -> int:
    return max_val + min_val - value


def mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


@dataclass
class ScaleSubscore:
    raw_mean: Optional[float]
    normalized_0_10: Optional[float]
    n_items: int


@dataclass
class Bat12Score:
    exhaustion: ScaleSubscore
    mental_distance: ScaleSubscore
    cognitive_impairment: ScaleSubscore
    emotional_impairment: ScaleSubscore
    total: ScaleSubscore
    item_count: int = 12

    def to_dict(self) -> dict:
        return {
            "scale": "BAT-12",
            "item_count": self.item_count,
            "subscales": {
                "exhaustion": self.exhaustion.__dict__,
                "mental_distance": self.mental_distance.__dict__,
                "cognitive_impairment": self.cognitive_impairment.__dict__,
                "emotional_impairment": self.emotional_impairment.__dict__,
            },
            "total": self.total.__dict__,
        }


@dataclass
class OlbiScore:
    exhaustion: ScaleSubscore
    disengagement: ScaleSubscore
    total: ScaleSubscore
    item_count: int = 16

    def to_dict(self) -> dict:
        return {
            "scale": "OLBI",
            "item_count": self.item_count,
            "subscales": {
                "exhaustion": self.exhaustion.__dict__,
                "disengagement": self.disengagement.__dict__,
            },
            "total": self.total.__dict__,
        }


BAT12_SUBSCALES = {
    "exhaustion": ["bat1", "bat2", "bat3"],
    "mental_distance": ["bat4", "bat5", "bat6"],
    "cognitive_impairment": ["bat7", "bat8", "bat9"],
    "emotional_impairment": ["bat10", "bat11", "bat12"],
}
BAT12_RANGE = (1, 5)

OLBI_EXHAUSTION_ITEMS = [
    "olbi2", "olbi4", "olbi5", "olbi8",
    "olbi10", "olbi12", "olbi14", "olbi16",
]
OLBI_DISENGAGEMENT_ITEMS = [
    "olbi1", "olbi3", "olbi6", "olbi7",
    "olbi9", "olbi11", "olbi13", "olbi15",
]
OLBI_REVERSE_ITEMS = {
    "olbi2", "olbi3", "olbi4", "olbi6",
    "olbi8", "olbi9", "olbi11", "olbi12",
}
OLBI_RANGE = (1, 4)


def _subscale_mean_from_dict(responses: Dict[str, int], item_ids: List[str], reverse_items=None, min_val=None, max_val=None) -> ScaleSubscore:
    reverse_items = reverse_items or set()
    vals = []
    for item_id in item_ids:
        v = responses.get(item_id)
        if v is None:
            continue
        if item_id in reverse_items:
            v = reverse_score(v, min_val, max_val)
        vals.append(v)

    raw = mean(vals)
    return ScaleSubscore(
        raw_mean=raw,
        normalized_0_10=normalize_mean_to_0_10(raw, min_val, max_val) if raw is not None else None,
        n_items=len(vals),
    )


def score_bat12(responses: Dict[str, int]) -> Bat12Score:
    exhaustion = _subscale_mean_from_dict(responses, BAT12_SUBSCALES["exhaustion"], min_val=1, max_val=5)
    mental_distance = _subscale_mean_from_dict(responses, BAT12_SUBSCALES["mental_distance"], min_val=1, max_val=5)
    cognitive_impairment = _subscale_mean_from_dict(responses, BAT12_SUBSCALES["cognitive_impairment"], min_val=1, max_val=5)
    emotional_impairment = _subscale_mean_from_dict(responses, BAT12_SUBSCALES["emotional_impairment"], min_val=1, max_val=5)

    total_raw = mean([
        exhaustion.raw_mean,
        mental_distance.raw_mean,
        cognitive_impairment.raw_mean,
        emotional_impairment.raw_mean,
    ])

    total = ScaleSubscore(
        raw_mean=total_raw,
        normalized_0_10=normalize_mean_to_0_10(total_raw, 1, 5) if total_raw is not None else None,
        n_items=sum(x.n_items for x in [exhaustion, mental_distance, cognitive_impairment, emotional_impairment]),
    )

    return Bat12Score(
        exhaustion=exhaustion,
        mental_distance=mental_distance,
        cognitive_impairment=cognitive_impairment,
        emotional_impairment=emotional_impairment,
        total=total,
    )


def score_olbi(responses: Dict[str, int]) -> OlbiScore:
    exhaustion = _subscale_mean_from_dict(
        responses,
        OLBI_EXHAUSTION_ITEMS,
        reverse_items=OLBI_REVERSE_ITEMS,
        min_val=1,
        max_val=4,
    )
    disengagement = _subscale_mean_from_dict(
        responses,
        OLBI_DISENGAGEMENT_ITEMS,
        reverse_items=OLBI_REVERSE_ITEMS,
        min_val=1,
        max_val=4,
    )

    total_raw = mean([exhaustion.raw_mean, disengagement.raw_mean])

    total = ScaleSubscore(
        raw_mean=total_raw,
        normalized_0_10=normalize_mean_to_0_10(total_raw, 1, 4) if total_raw is not None else None,
        n_items=exhaustion.n_items + disengagement.n_items,
    )

    return OlbiScore(
        exhaustion=exhaustion,
        disengagement=disengagement,
        total=total,
    )
