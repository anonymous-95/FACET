"""Create ``task_desc.json`` files for new tasks with the FACET prompt.

The prompt is fixed; the LLM that answers it is your choice. Two steps:

1. Build the prompt for one or more task folders, each holding a Patho-Bench
   style ``config.yaml``::

       python scripts/generate_descriptions.py prompt my_tasks/cohort/my_task --out prompt.txt

   Paste the contents of ``prompt.txt`` into any chat interface and save the
   answer, a JSON array with one object per task, to ``answer.json``.

2. Write the answer into the task folders, in the same order::

       python scripts/generate_descriptions.py parse answer.json my_tasks/cohort/my_task

For more tasks than you want to paste by hand, use
``facet.text.generate_descriptions(task_dirs, llm=...)``, which sends the same
prompt through an LLM API and writes every ``task_desc.json`` for you.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from facet.text.generate import build_prompt, load_task_config, parse_response


def cmd_prompt(args) -> int:
    configs = [load_task_config(d) for d in args.task_dirs]
    prompt = build_prompt(configs)
    if args.out:
        Path(args.out).write_text(prompt, encoding="utf-8")
        print(f"wrote {args.out} ({len(configs)} tasks)")
    else:
        print(prompt)
    return 0


def cmd_parse(args) -> int:
    configs = [load_task_config(d) for d in args.task_dirs]
    answer = Path(args.answer).read_text(encoding="utf-8")
    descs = parse_response(answer, n_tasks=len(configs), label_dicts=[c["label_dict"] for c in configs])
    for task_dir, desc in zip(args.task_dirs, descs):
        path = Path(task_dir) / "task_desc.json"
        if path.exists() and not args.overwrite:
            print(f"skip {path} (exists; use --overwrite)")
            continue
        desc.save(path)
        print(f"wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompt", help="build the prompt for task folders")
    p.add_argument("task_dirs", type=Path, nargs="+")
    p.add_argument("--out", default=None, help="write the prompt here instead of stdout")
    p.set_defaults(func=cmd_prompt)

    q = sub.add_parser("parse", help="write an LLM answer into task_desc.json files")
    q.add_argument("answer", help="file holding the answer (a JSON array)")
    q.add_argument("task_dirs", type=Path, nargs="+")
    q.add_argument("--overwrite", action="store_true")
    q.set_defaults(func=cmd_parse)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
