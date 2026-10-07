"""Your voice recordings: the training data for your own speech model.

Your data folder is encrypted (see vault.py):
    clips/<id>.wav.enc   16 kHz mono recordings
    index.enc            one row per clip: file, text, source, checked, seconds

A plain folder (clips/<id>.wav + metadata.csv) is what `better-voice-record --pack` makes for
training on another computer such as Colab, after you accept the sharing agreement. Folders
from older versions are plain too; private() encrypts them in place the first time.

`source` is "prompt" (you read a given sentence) or "dictation" (saved from real use).
`checked` is 1 when the text is exactly what was said. Prompt recordings start checked;
dictations start unchecked because the text is the model's guess. Fix and check them with
better-voice-record --review, then the next training run uses them.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import vault

FIELDS = ["file", "text", "source", "checked", "seconds"]


@dataclass
class Clip:
    file: str
    text: str
    source: str
    checked: bool
    seconds: float = 0.0


def private(root: Path, encrypt: bool = True) -> "Dataset":
    """Your own data folder, kept the way the `encrypt` setting says: data in the other form
    (older versions, or the setting just changed) is converted in place first."""
    from .audio import decrypt_model_dir, encrypt_model_dir

    Path(root).mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(root, 0o700)  # only you (Windows keeps AppData per-user already)
    if encrypt:
        ds = Dataset(root, encrypted=True)
        if ds.migrate():
            print(f"Encrypted your recordings in {root}.")
        if encrypt_model_dir(Path(root) / "voice-model"):
            print("Encrypted your trained model.")
        return ds
    ds = Dataset(root, encrypted=False)
    if ds.decrypt_in_place():
        print(f"Encryption is off: decrypted your recordings in {root} to plain files.")
    if decrypt_model_dir(Path(root) / "voice-model"):
        print("Encryption is off: decrypted your trained model.")
    return ds


class Dataset:
    def __init__(self, root: Path, encrypted: bool | None = None) -> None:
        self.root = Path(root)
        self.clips_dir = self.root / "clips"
        self.index = self.root / "index.enc"
        self.csv = self.root / "metadata.csv"
        # A folder holding only plain data (a pack) stays plain unless told otherwise.
        self.encrypted = encrypted if encrypted is not None else not self.csv.exists()

    # ----- the index -----

    def _all(self) -> list[Clip]:
        if self.encrypted:
            if not self.index.exists():
                return []
            return [Clip(**r) for r in json.loads(vault.decrypt(self.index.read_bytes()))]
        if not self.csv.exists():
            return []
        with self.csv.open(newline="", encoding="utf-8") as f:
            return [Clip(r["file"], r["text"].strip(), r.get("source", ""), r.get("checked", "0").strip() == "1",
                         float(r.get("seconds") or 0)) for r in csv.DictReader(f)]

    def _save(self, clips: list[Clip]) -> None:
        # Write a temporary file and swap it in, so a crash can never leave a half-written index.
        self.root.mkdir(parents=True, exist_ok=True)
        if self.encrypted:
            data = vault.encrypt(json.dumps([asdict(c) for c in clips]).encode())
            target = self.index
        else:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=FIELDS)
            w.writeheader()
            w.writerows({**asdict(c), "checked": int(c.checked)} for c in clips)
            data, target = buf.getvalue().encode(), self.csv
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)

    # ----- clips -----

    def rows(self, checked_only: bool = True) -> list[Clip]:
        return [c for c in self._all() if c.text and (c.checked or not checked_only) and (self.root / c.file).exists()]

    def seconds(self, checked_only: bool = True) -> float:
        return sum(c.seconds for c in self.rows(checked_only))

    def load(self, clip: Clip) -> np.ndarray:
        from .audio import load_audio

        raw = (self.root / clip.file).read_bytes()
        return load_audio(io.BytesIO(vault.decrypt(raw) if self.encrypted else raw))

    def add(self, audio: np.ndarray, text: str, source: str, checked: bool) -> Clip:
        from .audio import SAMPLE_RATE, wav_bytes

        self.clips_dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}.wav"
        data = wav_bytes(audio)
        if self.encrypted:
            name, data = name + ".enc", vault.encrypt(data)
        (self.clips_dir / name).write_bytes(data)
        clip = Clip(f"clips/{name}", text.strip(), source, checked, round(len(audio) / SAMPLE_RATE, 2))
        self._save(self._all() + [clip])
        return clip

    def update(self, file: str, text: str, checked: bool) -> None:
        self._save([Clip(c.file, text, c.source, checked, c.seconds) if c.file == file else c for c in self._all()])

    def remove(self, file: str) -> bool:
        """Delete one recording and its row (for "redo"). Returns False if it wasn't there."""
        clips = self._all()
        keep = [c for c in clips if c.file != file]
        if len(keep) == len(clips):
            return False
        self._save(keep)
        (self.root / file).unlink(missing_ok=True)
        return True

    # ----- moving between plain and encrypted -----

    def migrate(self) -> int:
        """Encrypt plain recordings left by older versions, in place. Returns how many."""
        if not self.encrypted or not self.csv.exists():
            return 0
        import soundfile as sf

        plain = Dataset(self.root, encrypted=False)
        moved = self._all()
        for c in plain._all():
            src = self.root / c.file
            if not src.exists():
                continue
            seconds = c.seconds or sf.info(str(src)).duration
            dst = src.with_name(src.name + ".enc")
            dst.write_bytes(vault.encrypt(src.read_bytes()))
            moved.append(Clip(dst.relative_to(self.root).as_posix(), c.text, c.source, c.checked, round(seconds, 2)))
            self._save(moved)  # saved after every clip, so an interruption loses nothing
            src.unlink()
        self.csv.unlink()
        return len(moved)

    def decrypt_in_place(self) -> int:
        """Turn encrypted recordings back into plain files (encryption switched off)."""
        if self.encrypted or not self.index.exists():
            return 0
        enc = Dataset(self.root, encrypted=True)
        moved = self._all()
        for c in enc._all():
            src = self.root / c.file
            if not src.exists():
                continue
            dst = src.with_name(src.name.removesuffix(".enc"))
            dst.write_bytes(vault.decrypt(src.read_bytes()))
            moved.append(Clip(dst.relative_to(self.root).as_posix(), c.text, c.source, c.checked, c.seconds))
            self._save(moved)  # saved after every clip, so an interruption loses nothing
            src.unlink()
        self.index.unlink()
        return len(moved)

    def export_plain(self, dest: Path) -> "Dataset":
        """Decrypted copy for training elsewhere (`--pack`, only after the sharing agreement)."""
        out = Dataset(dest, encrypted=False)
        clips = []
        for c in self.rows(checked_only=False):
            name = Path(c.file).name.removesuffix(".enc")
            (out.clips_dir).mkdir(parents=True, exist_ok=True)
            raw = (self.root / c.file).read_bytes()
            (out.clips_dir / name).write_bytes(vault.decrypt(raw) if self.encrypted else raw)
            clips.append(Clip(f"clips/{name}", c.text, c.source, c.checked, c.seconds))
        out._save(clips)
        return out
