"""Text cleaning applied before vectorisation.

The cleaner is deliberately simple and transparent: it normalises case,
replaces URLs and email addresses with placeholder tokens, strips digits from
order/ticket numbers, and collapses whitespace. It is a module-level function
so the fitted scikit-learn pipeline can be pickled and reloaded.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_ORDER_NUMBER_RE = re.compile(r"#?\b\d{3,}\b")
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s']")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return a normalised version of ``text`` for the classifier."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    cleaned = text.lower()
    cleaned = _URL_RE.sub(" urltoken ", cleaned)
    cleaned = _EMAIL_RE.sub(" emailtoken ", cleaned)
    cleaned = _ORDER_NUMBER_RE.sub(" numbertoken ", cleaned)
    cleaned = _NON_TEXT_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned
