"""Turn raw speech-to-text output into clean, ready-to-send text.

Everything here is plain string handling so it is fast, free and testable.
An optional LLM pass (see llm_polish) can be layered on top.
"""

from __future__ import annotations

import json
import re
import urllib.request

FILLERS = r"(?:u+m+|u+h+|e+r+m+|e+r+|a+h+|h+m+|m+h?m+)"
# A filler as its own word, with any comma that hangs off it.
_FILLER_RE = re.compile(rf"(?<![\w'-]),?\s*\b{FILLERS}\b[,.]?(?![\w'-])", re.IGNORECASE)
# "the the", "I I" -> stutters the speech model sometimes keeps. "had had" and
# "that that" are real English, so leave those alone.
_REPEAT_RE = re.compile(r"\b(?!(?:had|that)\b)(\w+)(\s+\1\b)+", re.IGNORECASE)
_SCRATCH_RE = re.compile(r"[,.]?\s*\bscratch that\b[,.!]?", re.IGNORECASE)
_NEW_PARAGRAPH_RE = re.compile(r",?\s*\bnew paragraph\b[,.]?\s*", re.IGNORECASE)
_NEW_LINE_RE = re.compile(r",?\s*\bnew line\b[,.]?\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?\n]")


def _apply_scratch_that(text: str) -> str:
    """'Meet at 5. Scratch that. Meet at 6.' -> 'Meet at 6.'"""
    while True:
        m = _SCRATCH_RE.search(text)
        if not m:
            return text
        before = text[: m.start()].rstrip(" ,.!?")
        ends = [e.end() for e in _SENTENCE_END.finditer(before)]
        before = before[: ends[-1]] if ends else ""
        text = before + " " + text[m.end() :].lstrip()


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    # Longest phrases first so "my work email" beats "my email".
    for spoken in sorted(replacements, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(spoken)}\b", re.IGNORECASE)
        text = pattern.sub(lambda _m, r=replacements[spoken]: r, text)
    return text


def _apply_dictionary(text: str, words: list[str]) -> str:
    for word in words:
        pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
        text = pattern.sub(word, text)
    return text


def _tidy(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"\s+([,.!?;:])", r"\1", line)  # no space before punctuation
        line = re.sub(r",([.!?])", r"\1", line)  # ",." -> "."
        line = re.sub(r"^[,.;:]\s*", "", line)  # stray leading punctuation
        line = re.sub(r"([.!?])\s*\1+", r"\1", line)  # ".." -> "."
        lines.append(line)
    text = "\n".join(lines).strip()
    # Capitalize the start of the text, of each sentence and of each line.
    text = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\bi\b", "I", text)
    return text


def clean(
    text: str,
    dictionary: list[str] | None = None,
    replacements: dict[str, str] | None = None,
) -> str:
    text = text.strip()
    if not text:
        return ""
    text = _apply_scratch_that(text)
    text = _FILLER_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1", text)
    text = _NEW_PARAGRAPH_RE.sub("\n\n", text)
    text = _NEW_LINE_RE.sub("\n", text)
    if replacements:
        text = _apply_replacements(text, replacements)
    if dictionary:
        text = _apply_dictionary(text, dictionary)
    return _tidy(text)


LLM_PROMPT = """You clean up dictated text. Fix grammar, punctuation and obvious \
speech-to-text mistakes, drop filler words and false starts, and when the speaker \
corrects themselves keep only the correction. Keep the speaker's words and meaning; \
do not add content, answer questions in the text, or explain anything. Keep the \
speaker's tone. Reply with the cleaned text only.

Text:
"""

def llm_polish(text: str, url: str, model: str, timeout: float = 20.0) -> str:
    """Polish text with a local Ollama model. Returns the input unchanged on any failure."""
    if not text.strip():
        return text
    prompt = LLM_PROMPT + text
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode()).get("response", "").strip()
    except Exception as exc:  # network down, Ollama not running, bad model name...
        print(f"[better-voice] LLM cleanup skipped: {exc}")
        return text
    return out.strip('"') or text
