"""Patient-level clinical context appended to task descriptions.

FACET can take clinical context at inference time with no retraining. A short
sentence describing a patient attribute is appended to the task description,
and the augmented text is embedded per patient.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Mapping, Optional, Union

from .descriptions import TaskDescription

UNKNOWN_KEY = "Unknown"


def parse_format_map(format_map: Union[str, Mapping[str, str]]) -> Dict[str, str]:
    """Parse a mapping (or its JSON string) from raw clinical values to sentences."""
    if isinstance(format_map, str):
        format_map = json.loads(format_map)
    if not isinstance(format_map, Mapping) or not format_map:
        raise ValueError("format map must be a non-empty JSON object")
    return {str(k): str(v) for k, v in format_map.items()}


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip() or str(value).strip().lower() == "nan"


def clinical_sentence(value, format_map: Mapping[str, str]) -> Optional[str]:
    """Map a raw clinical value to its sentence.

    Missing values map to ``format_map["Unknown"]`` when that key exists and to
    ``None`` (no augmentation) otherwise. Unmapped present values raise.
    """
    if _is_missing(value):
        return format_map.get(UNKNOWN_KEY)
    key = str(value).strip()
    if key not in format_map:
        raise ValueError(f"clinical value {key!r} not in format map (keys: {sorted(format_map)})")
    return format_map[key]


def augment_task_text(base_task: str, value, format_map: Mapping[str, str]) -> str:
    """Append the sentence for ``value`` to a task description string."""
    sentence = clinical_sentence(value, format_map)
    if sentence is None:
        return base_task
    sentence = sentence.strip()
    if not sentence.endswith("."):
        sentence += "."
    return f"{base_task} {sentence}"


def augment_task_description(
    desc: TaskDescription, value, format_map: Mapping[str, str]
) -> TaskDescription:
    """Return a copy of ``desc`` whose task text carries the clinical context.

    Class descriptions are left unchanged.
    """
    return TaskDescription(task=augment_task_text(desc.task, value, format_map), classes=dict(desc.classes))
