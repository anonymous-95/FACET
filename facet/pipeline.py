"""High-level inference: patch features + task description -> ``z_t``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from .io.features import load_patch_features
from .modeling import FACET
from .text.descriptions import TaskDescription, as_task_description
from .text.encoder import TextEncoder, load_text_encoder

PathLike = Union[str, Path]
SlideInput = Union[PathLike, Tuple[torch.Tensor, torch.Tensor]]
TaskInput = Union[TaskDescription, Mapping, PathLike, str]


def _is_existing_path(text: str) -> bool:
    """True if ``text`` names an existing file or directory.

    Task descriptions are sentences, not paths. A single path component longer
    than 255 bytes is not a valid filename (``NAME_MAX``), and ``Path.exists``
    raises ``OSError: [Errno 36] File name too long`` instead of returning
    False. Treat those strings as raw text.
    """
    try:
        path = Path(text)
        if any(len(part) > 255 for part in path.parts):
            return False
        return path.exists()
    except (OSError, ValueError):
        return False


class FACETPipeline:
    """Bundles FACET with its text encoder.

    Example::

        pipe = FACETPipeline.from_pretrained("path/to/facet")   # or a hub id
        z = pipe.embed_slide("slide.h5", task="data/evaluation/cptac_coad/KRAS_mutation")
        Z = pipe.embed_slides(["a.h5", "b.h5"], task=...)       # same, for many slides
    """

    def __init__(self, model: FACET, text_encoder: TextEncoder, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.text_encoder = text_encoder
        self._text_cache: Dict[str, torch.Tensor] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: PathLike,
        text_encoder_checkpoint: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        hf_token: Optional[str] = None,
        **hub_kwargs,
    ) -> "FACETPipeline":
        """Load FACET weights and its text encoder.

        Args:
            model_name_or_path: HuggingFace repo id or a local directory with
                ``config.json`` and ``model.safetensors``.
            text_encoder_checkpoint: local CONCH ``pytorch_model.bin``; defaults
                to the (gated) HuggingFace copy.
            device: torch device.
            hf_token: optional HuggingFace token for gated downloads.
            **hub_kwargs: forwarded to ``FACET.from_pretrained``
                (e.g. ``revision``, ``local_files_only``).
        """
        model = FACET.from_pretrained(model_name_or_path, token=hf_token, **hub_kwargs)
        text_encoder = load_text_encoder(
            model.config.text_encoder, checkpoint=text_encoder_checkpoint, device=device, hf_token=hf_token
        )
        return cls(model, text_encoder, device=device)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        """Text-encoder embedding (text_dim,) of a single string, cached."""
        if text not in self._text_cache:
            self._text_cache[text] = self.text_encoder.encode([text])[0].to(self.device)
        return self._text_cache[text]

    def task_text(self, task: TaskInput) -> str:
        """Resolve a task input to its description string.

        ``task`` may be a :class:`TaskDescription`, a dict, a path to a
        ``task_desc.json`` / task folder, or the raw description text.
        """
        if isinstance(task, (TaskDescription, Mapping)):
            return as_task_description(task).task
        if isinstance(task, Path) or (isinstance(task, str) and _is_existing_path(task)):
            return as_task_description(task).task
        return str(task)

    @torch.no_grad()
    def embed_slide(self, slide: SlideInput, task: TaskInput) -> torch.Tensor:
        """Task-conditioned embedding ``z_t`` (embed_dim,) of one slide.

        Args:
            slide: path to a TRIDENT ``.h5`` feature file, or a
                ``(features (N, D), grid_coords (N, 2))`` tuple.
            task: task description (see :meth:`task_text`).
        """
        if isinstance(slide, (str, Path)):
            features, coords = load_patch_features(slide)
        else:
            features, coords = slide
        task_emb = self.embed_text(self.task_text(task))
        return self.model(
            features.to(self.device).float(), coords.to(self.device).long(), task_emb
        )

    @torch.no_grad()
    def embed_slides(self, slides: Sequence[SlideInput], task: TaskInput) -> torch.Tensor:
        """Task-conditioned embeddings ``(len(slides), embed_dim)`` for several slides.

        Slides are padded to a common length and embedded in one forward pass, so
        memory grows with the largest slide in the batch. Pass a single task, or
        call this once per task.
        """
        loaded = [load_patch_features(s) if isinstance(s, (str, Path)) else s for s in slides]
        n = max(f.shape[0] for f, _ in loaded)
        features = torch.zeros(len(loaded), n, self.model.config.input_dim)
        coords = torch.zeros(len(loaded), n, 2, dtype=torch.long)
        attn_mask = torch.zeros(len(loaded), n, dtype=torch.bool)
        for i, (f, c) in enumerate(loaded):
            features[i, : f.shape[0]] = f
            coords[i, : c.shape[0]] = c
            attn_mask[i, : f.shape[0]] = True
        task_emb = self.embed_text(self.task_text(task))
        return self.model(
            features.to(self.device),
            coords.to(self.device),
            task_emb,
            attn_mask=attn_mask.to(self.device),
        )
