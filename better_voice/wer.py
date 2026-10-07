"""Word error rate: the share of words a model gets wrong. Lower is better; 0.05 means 1 in 20."""

from __future__ import annotations

import re


def normalize(text: str) -> list[str]:
    """Compare words, not formatting: lowercase, no punctuation except inside words (it's, 4:30)."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w'$:.%-]+", " ", text)
    words = [w.strip(".'-:") for w in text.split()]
    return [w for w in words if w]


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    errors = total = 0
    for r, h in zip(refs, hyps):
        rw = normalize(r)
        errors += edits(rw, normalize(h))
        total += len(rw)
    return errors / total if total else 0.0
