"""User settings, stored as JSON in the user's config directory."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def user_dir(data: bool = False) -> Path:
    """The per-user app folder root on this OS (Linux splits settings and data)."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if data:
        return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def config_dir() -> Path:
    return user_dir() / "better-voice"


def data_dir() -> Path:
    """Voice recordings and your trained model live here."""
    return config_dir() / "voice-data"


@dataclass
class Config:
    # Speech model: the open Moonshine base model, or the folder of your own trained copy
    # (better-voice-train writes one to voice-model/ in the data folder).
    model: str = "UsefulSensors/moonshine-base"
    device: str = "auto"  # auto, cpu, cuda, mps
    # Rule-based cleanup: filler words, punctuation, capitals, spoken commands.
    cleanup: bool = True
    # Optional extra polish by a local LLM through Ollama (free, runs on your machine).
    llm_cleanup: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    # Words to spell your way (names, jargon). Fixes their capitals after transcription;
    # training on your voice is what teaches the model to hear them.
    dictionary: list[str] = field(default_factory=list)
    # Spoken phrase -> replacement, e.g. {"my email": "me@example.com"}.
    replacements: dict[str, str] = field(default_factory=dict)
    # Keep a local log of everything dictated.
    history: bool = True
    # Save each dictation's audio next to its text, so you can correct the text and train on it.
    keep_audio: bool = False
    # Encrypt your recordings and trained model on this computer (key in the OS credential
    # store). Turning it off decrypts them back to plain files the next time a command runs.
    encrypt: bool = True
    # Password for `better-voice --serve`; created on first start.
    server_token: str = ""
    # When you accepted the agreement for sending voice data off this computer (consent.py).
    sharing_agreement: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or config_dir() / "config.json"
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            print(f"[better-voice] ignoring unknown settings: {', '.join(sorted(unknown))}")
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        path = path or config_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
