"""Official implementation of the Ghostwriter attack."""

from .core import (
    Candidate,
    RepackagingResult,
    inject_statement,
    repackage_statement,
)
from .models import ChatModel, Message

__all__ = [
    "Candidate",
    "ChatModel",
    "Message",
    "RepackagingResult",
    "inject_statement",
    "repackage_statement",
]

__version__ = "0.1.0"
