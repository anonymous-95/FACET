"""Cache CONCH text embeddings of task descriptions to a ``.safetensors`` file.

Useful to encode descriptions once (e.g. on a machine with CONCH access) and
reuse them without loading the text encoder. Keys are ``<dataset>/<task>``;
values are (512,) task-description embeddings. Use them as ``task_embedding``
in ``FACET.forward``.

Example::

    python scripts/embed_descriptions.py --tasks-root data/evaluation --out task_embeddings.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from facet.text import TaskDescription, load_text_encoder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-encoder-checkpoint", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths = sorted(args.tasks_root.glob("*/*/task_desc.json"))
    if not paths:
        ap.error(f"no task_desc.json under {args.tasks_root}")
    keys = [f"{p.parent.parent.name}/{p.parent.name}" for p in paths]
    texts = [TaskDescription.load(p).task for p in paths]

    encoder = load_text_encoder(checkpoint=args.text_encoder_checkpoint, device=args.device)
    with torch.no_grad():
        embs = torch.cat([encoder.encode(texts[i : i + 32]) for i in range(0, len(texts), 32)])
    save_file({k: e.cpu().contiguous() for k, e in zip(keys, embs)}, str(args.out))
    print(f"wrote {len(keys)} task embeddings to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
