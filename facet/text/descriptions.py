"""Task / class description schema (``task_desc.json``).

A task description file looks like::

    {
      "task": "<1-3 sentence description of the prediction objective>",
      "classes": {"0": "<class 0 description>", "1": "<class 1 description>"}
    }

Class keys are string integer indices matching the task's ``label_dict``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

PathLike = Union[str, Path]
TASK_DESC_FILENAME = "task_desc.json"


@dataclass
class TaskDescription:
    task: str
    classes: Dict[str, str] = field(default_factory=dict)

    @property
    def class_texts(self) -> List[str]:
        """Class descriptions ordered by integer class index."""
        return [self.classes[k] for k in sorted(self.classes, key=int)]

    def with_context(self, context: str) -> "TaskDescription":
        """Return a copy whose task text has ``context`` appended."""
        return TaskDescription(task=f"{self.task} {context}".strip(), classes=dict(self.classes))

    def validate(self, label_dict: Optional[Mapping] = None) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("'task' must be a non-empty string")
        if not isinstance(self.classes, dict):
            raise ValueError("'classes' must be a mapping")
        for k, v in self.classes.items():
            if not str(k).isdigit():
                raise ValueError(f"class key {k!r} is not an integer index")
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"class {k!r} has an empty description")
        if label_dict is not None:
            expected = {str(k) for k in label_dict}
            if set(self.classes) != expected:
                raise ValueError(
                    f"class keys {sorted(self.classes, key=int)} do not match "
                    f"label_dict keys {sorted(expected, key=int)}"
                )

    def to_dict(self) -> Dict:
        return {"task": self.task, "classes": {k: self.classes[k] for k in sorted(self.classes, key=int)}}

    @classmethod
    def from_dict(cls, d: Mapping) -> "TaskDescription":
        if "task" not in d:
            raise ValueError("task description must contain a 'task' field")
        return cls(task=d["task"], classes={str(k): v for k, v in (d.get("classes") or {}).items()})

    @classmethod
    def load(cls, path: PathLike) -> "TaskDescription":
        """Load from a ``task_desc.json`` file or from a task folder containing one."""
        path = Path(path)
        if path.is_dir():
            path = path / TASK_DESC_FILENAME
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / TASK_DESC_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")
        return path


def as_task_description(task: Union[TaskDescription, PathLike, Mapping]) -> TaskDescription:
    """Accept a TaskDescription, a dict, a ``task_desc.json`` path, or a task folder."""
    if isinstance(task, TaskDescription):
        return task
    if isinstance(task, Mapping):
        return TaskDescription.from_dict(task)
    return TaskDescription.load(task)
