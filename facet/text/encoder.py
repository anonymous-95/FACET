"""Text encoders producing the task / class embeddings consumed by FACET."""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Union

import torch

CONCH_HF_ID = "hf_hub:MahmoodLab/conch"


class TextEncoder(Protocol):
    """Anything that maps a list of strings to a (len(texts), dim) tensor."""

    dim: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class CONCHTextEncoder:
    """CONCH v1 text tower (512-d), the text encoder FACET was trained with.

    Weights are gated: request access at https://huggingface.co/MahmoodLab/conch
    and either log in (``huggingface-cli login``) or pass a local
    ``pytorch_model.bin`` via ``checkpoint``.

    Args:
        checkpoint: ``"hf_hub:MahmoodLab/conch"`` (default) or a local path to
            the CONCH ``pytorch_model.bin``.
        device: torch device for the text tower.
        hf_token: optional HuggingFace access token.
    """

    dim = 512

    def __init__(
        self,
        checkpoint: str = CONCH_HF_ID,
        device: Optional[Union[str, torch.device]] = None,
        hf_token: Optional[str] = None,
    ):
        try:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        except ImportError as e:
            raise ImportError(
                "CONCH is required for text encoding: "
                "pip install git+https://github.com/Mahmoodlab/CONCH.git"
            ) from e

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        kwargs = {"hf_auth_token": hf_token} if hf_token else {}
        model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint, **kwargs)
        self.model = model.to(self.device).eval()
        self.tokenizer = get_tokenizer()

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Embed texts; returns a (len(texts), 512) float32 tensor on ``self.device``."""
        from conch.open_clip_custom import tokenize

        tokens = tokenize(texts=list(texts), tokenizer=self.tokenizer).to(self.device)
        return self.model.encode_text(tokens).float()


def load_text_encoder(
    name: str = "conch_v1",
    checkpoint: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    hf_token: Optional[str] = None,
) -> TextEncoder:
    """Instantiate the text encoder named in ``FACETConfig.text_encoder``."""
    if name != "conch_v1":
        raise ValueError(f"Unsupported text encoder {name!r}; FACET uses 'conch_v1'.")
    return CONCHTextEncoder(checkpoint or CONCH_HF_ID, device=device, hf_token=hf_token)


def encode_texts(encoder: TextEncoder, texts: Union[str, List[str]]) -> torch.Tensor:
    if isinstance(texts, str):
        texts = [texts]
    return encoder.encode(texts)
