"""Text normalization for Slack messages."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for embedding input.

    Steps:
    1. Unicode NFKC normalization (handles fullwidth → halfwidth for alphanumeric)
    2. Compress consecutive whitespace/newlines to single space
    3. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
