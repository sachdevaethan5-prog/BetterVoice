"""Sentences to read aloud when recording training data.

The built-in list covers what you actually say to a voice assistant and in dictation:
commands, questions, texts, emails, dates, times, money and numbers (data/everyday.txt),
plus 1,800 general sentences (data/sentences.txt). Your own words are
mixed in from the `dictionary` setting, and any sentences you put in prompts.txt in the
data folder (one per line) come first. Personal details stay in those local files, not here.
"""

from __future__ import annotations

import random
from pathlib import Path

TEMPLATES = [
    "Remind me to talk to {w} tomorrow.",
    "Did {w} say anything about it?",
    "Add {w} to my notes.",
    "What's the latest on {w}?",
    "I need to email {w} before Friday.",
    "Can you check {w} for me?",
    "I'm heading to {w} after class.",
    "Make sure {w} is spelled right.",
]


DATA = Path(__file__).parent / "data"


def _read(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def build(data_dir: Path, words: list[str], seed: int = 7) -> list[str]:
    """About 2,100 sentences, most useful first: your own prompts, sentences with your words,
    everyday assistant and texting sentences, then 1,800 general sentences (Mozilla Common Voice,
    public domain) for reading practice in your voice."""
    own = _read(Path(data_dir) / "prompts.txt")
    rng = random.Random(seed)
    with_words = [t.format(w=w) for w in words for t in rng.sample(TEMPLATES, k=min(3, len(TEMPLATES)))]
    everyday = _read(DATA / "everyday.txt")
    general = _read(DATA / "sentences.txt")
    rng.shuffle(with_words)
    rng.shuffle(everyday)
    # Mix a general sentence in after every two everyday ones, so a session isn't all commands.
    mixed, g = [], iter(general)
    for i, s in enumerate(everyday):
        mixed.append(s)
        if i % 2 == 1:
            mixed.append(next(g, ""))
    mixed.extend(g)
    seen, out = set(), []
    for s in own + with_words + mixed:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out
