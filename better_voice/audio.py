"""Microphone recording and on-device speech-to-text."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # what the speech model expects
# Moonshine is trained on clips up to about 30 seconds; longer audio is cut into pieces.
CHUNK_SECONDS = 25


class Recorder:
    """Records mono 16 kHz audio between start() and stop()."""

    def __init__(self) -> None:
        import sounddevice as sd

        self._sd = sd
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None

    def _callback(self, indata, frames, time, status) -> None:
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def start(self) -> None:
        with self._lock:
            self._chunks = []
        self._stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._callback
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


def load_audio(source) -> np.ndarray:
    """Read a WAV/FLAC/OGG file (path or file object) as mono float32 at 16 kHz."""
    import soundfile as sf

    audio, rate = sf.read(str(source) if isinstance(source, (str, Path)) else source, dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), rate)


def resample(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE or len(audio) == 0:
        return audio.astype(np.float32)
    n = int(round(len(audio) * SAMPLE_RATE / rate))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def wav_bytes(audio: np.ndarray) -> bytes:
    """16 kHz 16-bit mono WAV, in memory (so it can be encrypted before it touches the disk)."""
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def level(audio: np.ndarray) -> float:
    return float(np.abs(audio).max()) if len(audio) else 0.0


def prepare(audio: np.ndarray) -> np.ndarray:
    """Trim the silence at both ends and bring quiet recordings up to a steady level.

    The speech model does much worse on quiet audio, and laptop mics often record quietly.
    """
    frame = SAMPLE_RATE // 50  # 20 ms
    n = len(audio) // frame
    if n < 5:
        return audio
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.nonzero(rms > max(0.002, rms.max() * 0.1))[0]
    if len(loud):
        pad = 12  # keep 0.25 s around the speech
        audio = audio[max(0, (loud[0] - pad) * frame) : min(len(audio), (loud[-1] + 1 + pad) * frame)]
    peak = level(audio)
    if 0.003 < peak < 0.6:
        audio = audio * min(0.6 / peak, 30.0)
    return audio.astype(np.float32)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


WEIGHTS_ENC = "model.safetensors.enc"  # your trained model's weights, encrypted (see vault.py)


def load_moonshine(model: str):
    """(processor, model) for a Hugging Face name or a model folder; encrypted weights are
    decrypted in memory only."""
    from transformers import AutoConfig, AutoProcessor, MoonshineForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model)
    enc = Path(model) / WEIGHTS_ENC
    if not enc.exists():
        return processor, MoonshineForConditionalGeneration.from_pretrained(model)
    from safetensors.torch import load

    from . import vault

    m = MoonshineForConditionalGeneration(AutoConfig.from_pretrained(model))
    m.load_state_dict(load(vault.decrypt(enc.read_bytes())))
    return processor, m


def save_private(model, processor, out: Path) -> None:
    """Save a model with its weights encrypted; no plain copy of the weights touches the disk."""
    from safetensors.torch import save

    from . import vault

    out.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(out)
    model.generation_config.save_pretrained(out)
    processor.save_pretrained(out)
    # clone(): tied weights share memory, which safetensors refuses to save twice.
    weights = save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    (out / WEIGHTS_ENC).write_bytes(vault.encrypt(weights))
    (out / "model.safetensors").unlink(missing_ok=True)


def encrypt_model_dir(path: Path) -> bool:
    """Encrypt a plain trained model in place (from older versions or another computer)."""
    if not (Path(path) / "model.safetensors").exists():
        return False
    processor, model = load_moonshine(str(path))
    save_private(model, processor, Path(path))
    return True


def decrypt_model_dir(path: Path) -> bool:
    """Turn an encrypted trained model back into a plain one (encryption switched off)."""
    if not (Path(path) / WEIGHTS_ENC).exists():
        return False
    processor, model = load_moonshine(str(path))
    model.save_pretrained(path)
    processor.save_pretrained(path)
    (Path(path) / WEIGHTS_ENC).unlink()
    return True


class Transcriber:
    """Speech to text with a Moonshine model: the open base model, or your own fine-tuned copy.

    `model` is a Hugging Face name (UsefulSensors/moonshine-base) or a local folder written
    by `better-voice-train`.
    """

    def __init__(self, model: str, device: str = "auto") -> None:
        import torch
        from transformers.utils import logging

        logging.set_verbosity_error()  # hides a harmless max_length notice on every dictation
        self._torch = torch
        self.device = pick_device(device)
        self.processor, m = load_moonshine(model)
        self.model = m.to(self.device).eval()

    def transcribe(self, audio: np.ndarray) -> str:
        audio = prepare(audio)
        step = CHUNK_SECONDS * SAMPLE_RATE
        pieces = [audio[i : i + step] for i in range(0, len(audio), step)]
        return " ".join(t for t in (self._one(p) for p in pieces if len(p) > SAMPLE_RATE // 10) if t)

    def _one(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        # About 6.5 tokens per second of speech is plenty; the cap stops runaway repetition.
        limit = int(len(audio) / SAMPLE_RATE * 6.5) + 10
        with self._torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=limit)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
