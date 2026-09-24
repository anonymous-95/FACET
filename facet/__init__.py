"""FACET: task-conditioned whole-slide foundation model."""

from .config import FACETConfig
from .modeling import FACET
from .pipeline import FACETPipeline
from .text import TaskDescription, load_text_encoder

__version__ = "0.1.0"

__all__ = ["FACET", "FACETConfig", "FACETPipeline", "TaskDescription", "load_text_encoder"]
