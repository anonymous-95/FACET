"""Generating task / class descriptions for new tasks.

FACET's descriptions were produced with a fixed prompt
(``prompts/description_generation.md``) conditioned only on each task's
``config.yaml``. This module builds that prompt and parses the answer; which LLM
answers it is up to you.

For one or two tasks, build the prompt and paste it into a chat interface
(``scripts/generate_descriptions.py``). For many tasks, :func:`generate_descriptions`
runs the same prompt through an LLM API of your choice and writes a
``task_desc.json`` into every task folder::

    generate_descriptions(task_dirs, llm=my_api_call, batch_size=5)

``my_api_call`` is any ``(prompt: str) -> str`` callable, so no provider SDK is
part of FACET. Tasks are sent in small batches because the prompt asks for one
JSON object per task, and answers are validated against each task's ``label_dict``
before anything is written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from .descriptions import TaskDescription

PROMPT_PATH = Path(__file__).parent / "prompts" / "description_generation.md"

LLMCallable = Callable[[str], str]


def load_prompt(path: Union[str, Path] = PROMPT_PATH) -> str:
    """Return the fixed instruction prompt, without any task appended."""
    return Path(path).read_text(encoding="utf-8").strip()


def load_task_config(task_dir: Union[str, Path]) -> Dict:
    with open(Path(task_dir) / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_task_config(config: Mapping, index: int) -> str:
    body = yaml.dump(dict(config), default_flow_style=False, sort_keys=False).strip()
    return f"Task {index}:\n```yaml\n{body}\n```"


def build_prompt(configs: Sequence[Mapping], prompt_path: Union[str, Path] = PROMPT_PATH) -> str:
    """Build the complete prompt for a batch of task configs.

    The result is ready to paste into a chat interface or send to an API as a
    single message.
    """
    blocks = [format_task_config(c, i) for i, c in enumerate(configs, start=1)]
    return load_prompt(prompt_path) + "\n\n" + "\n\n".join(blocks) + "\n"


def _extract_json_array(text: str) -> Optional[list]:
    text = (text or "").strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
    return None


def parse_response(
    text: str,
    n_tasks: Optional[int] = None,
    label_dicts: Optional[Sequence[Mapping]] = None,
) -> List[TaskDescription]:
    """Parse the answer (a JSON array, optionally fenced) into descriptions.

    Args:
        text: the raw answer.
        n_tasks: expected number of descriptions.
        label_dicts: one ``label_dict`` per task; when given, each description's
            class keys must match, so a truncated answer fails here rather than
            silently producing a bad description.
    """
    items = _extract_json_array(text)
    if items is None:
        raise ValueError("could not find a JSON array in the response")
    if n_tasks is not None and len(items) != n_tasks:
        raise ValueError(f"expected {n_tasks} descriptions, got {len(items)}")
    descs = [TaskDescription.from_dict(item) for item in items]
    for i, d in enumerate(descs):
        d.validate(label_dicts[i] if label_dicts is not None else None)
    return descs


def generate_descriptions(
    task_dirs: Sequence[Union[str, Path]],
    llm: LLMCallable,
    batch_size: int = 5,
    overwrite: bool = False,
    prompt_path: Union[str, Path] = PROMPT_PATH,
) -> List[Path]:
    """Write a ``task_desc.json`` into each task folder, using an LLM API.

    This is the batch counterpart to pasting the prompt into a chat interface:
    the same fixed prompt is sent for ``batch_size`` tasks at a time and the
    answers are validated before being written.

    Args:
        task_dirs: folders, each holding a Patho-Bench style ``config.yaml``.
        llm: any ``(prompt: str) -> str`` callable wrapping your API of choice.
            We used a temperature of 0.
        batch_size: tasks per request.
        overwrite: regenerate descriptions that already exist.

    Returns:
        The paths written.
    """
    pending = [Path(d) for d in task_dirs if overwrite or not (Path(d) / "task_desc.json").exists()]
    written: List[Path] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        configs = [load_task_config(d) for d in batch]
        descs = parse_response(
            llm(build_prompt(configs, prompt_path)),
            n_tasks=len(batch),
            label_dicts=[c["label_dict"] for c in configs],
        )
        for d, desc in zip(batch, descs):
            written.append(desc.save(d / "task_desc.json"))
    return written
