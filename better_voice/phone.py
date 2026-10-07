"""Record training sentences with your phone's microphone; the clips land on this computer.

    better-voice-record --phone

Starts a small recording page on this computer and opens a free Cloudflare quick tunnel to
it, because phone browsers only allow the microphone on https pages. Open the printed link
(or scan the QR code) on your phone. Each sentence you record is saved straight into your
data folder, exactly like recordings made with the computer's mic. The link contains a
random secret and stops working when you press Ctrl+C here.

Needs `cloudflared` (free, no account): Windows `winget install --id Cloudflare.cloudflared`,
macOS `brew install cloudflared`, Linux: https://pkg.cloudflare.com
"""

from __future__ import annotations

import io
import re
import secrets
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .audio import SAMPLE_RATE, level, load_audio
from .dataset import Dataset
from .prompts import build
from .serve import Handler

PAGE = Path(__file__).parent / "data" / "phone.html"
MAX_BYTES = 4_000_000  # about 2 minutes of 16 kHz WAV; a sentence is a few seconds


class Session:
    """Which sentences are left. Shared by the request threads, so every change takes the lock."""

    def __init__(self, ds: Dataset, words: list[str]) -> None:
        self.ds = ds
        self.lock = threading.Lock()
        self.order = build(ds.root, words)
        self.done = {c.text.lower() for c in ds.rows(checked_only=False)}
        self.skipped: set[str] = set()
        self.minutes = ds.seconds() / 60

    def state(self) -> dict:
        nxt = next((s for s in self.order if s.lower() not in self.done and s.lower() not in self.skipped), "")
        left = sum(1 for s in self.order if s.lower() not in self.done)
        return {"sentence": nxt, "minutes": round(self.minutes, 2), "left": left}

    def save(self, audio: np.ndarray, sentence: str) -> dict:
        secs = len(audio) / SAMPLE_RATE
        peak = level(audio)
        if peak < 0.003:
            raise ValueError("The phone's mic picked up nothing. Check that the page is allowed to use it.")
        # Trim the tap noise at both ends, like the computer recorder does.
        cut = int(0.1 * SAMPLE_RATE)
        clip = self.ds.add(audio[cut:-cut] if len(audio) > 4 * cut else audio, sentence, "prompt", True)
        with self.lock:
            self.done.add(sentence.lower())
            self.minutes += secs / 60
        note = f"Saved, but quiet (level {peak:.3f}). Hold the phone a little closer." if peak < 0.03 else ""
        return {"file": clip.file, "seconds": secs, "note": note}

    def undo(self, file: str) -> None:
        rows = {c.file: c for c in self.ds.rows(checked_only=False)}
        clip = rows.get(file)
        if not clip or not self.ds.remove(file):
            raise ValueError("That recording is already gone.")
        with self.lock:
            self.done.discard(clip.text.lower())
            self.minutes = self.ds.seconds() / 60


def make_handler(session: Session, secret: str):
    page = PAGE.read_bytes()
    prefix = f"/{secret}/"

    class Recorder(Handler):
        def _route(self) -> str | None:
            # Anything without the secret gets a plain 404, so the tunnel URL alone reveals nothing.
            if self.path == prefix[:-1]:
                return ""
            return self.path[len(prefix):] if self.path.startswith(prefix) else None

        def do_GET(self) -> None:
            route = self._route()
            if route is None:
                return self._send(404, b"Not found", "text/plain")
            if route == "":
                return self._send(200, page, "text/html; charset=utf-8")
            if route == "state":
                return self._json(200, session.state())
            self._send(404, b"Not found", "text/plain")

        def do_POST(self) -> None:
            route = self._route()
            if route is None:
                return self._send(404, b"Not found", "text/plain")
            try:
                if route == "clip":
                    size = int(self.headers.get("Content-Length") or 0)
                    if not 44 < size <= MAX_BYTES:
                        return self._json(413 if size else 400, {"error": "That recording was empty or too long."})
                    sentence = unquote(self.headers.get("X-Sentence", "")).strip()
                    if not sentence:
                        return self._json(400, {"error": "The sentence was missing. Reload the page."})
                    audio = load_audio(io.BytesIO(self.rfile.read(size)))
                    result = session.save(audio, sentence)
                    return self._json(200, {**result, "state": session.state()})
                if route == "skip":
                    with session.lock:
                        session.skipped.add(unquote(self.headers.get("X-Sentence", "")).strip().lower())
                    return self._json(200, session.state())
                if route == "undo":
                    session.undo(self.headers.get("X-File", ""))
                    return self._json(200, {"state": session.state()})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # unreadable audio and the like
                return self._json(400, {"error": f"Couldn't save that recording: {exc}"})
            self._send(404, b"Not found", "text/plain")

        def log_message(self, fmt, *args) -> None:
            pass  # the tunnel URL is in the request line; keep it out of the terminal

    return Recorder


def open_tunnel(port: int, timeout: float = 45.0) -> tuple[subprocess.Popen, str]:
    """Start a Cloudflare quick tunnel to localhost and return (process, https URL)."""
    exe = shutil.which("cloudflared")
    if not exe:
        raise FileNotFoundError(
            "cloudflared isn't installed. It's free and needs no account:\n"
            "  Windows: winget install --id Cloudflare.cloudflared\n"
            "  macOS:   brew install cloudflared\n"
            "Then open a new terminal and run better-voice-record --phone again."
        )
    proc = subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    found: list[str] = []

    def read() -> None:
        for line in proc.stderr:  # cloudflared logs to stderr
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and not found:
                found.append(m.group(0))

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout)
    if not found:
        proc.terminate()
        raise RuntimeError("The Cloudflare tunnel didn't start. Check the internet connection and try again.")
    return proc, found[0]


def show_qr(url: str) -> None:
    try:
        import qrcode
    except ImportError:
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.print_ascii(invert=True)


def run(ds: Dataset, words: list[str], port: int = 8766, tunnel: bool = True) -> None:
    session = Session(ds, words)
    secret = secrets.token_urlsafe(18)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, secret))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proc = None
    try:
        if tunnel:
            proc, public = open_tunnel(server.server_port)
            url = f"{public}/{secret}/"
        else:
            url = f"http://127.0.0.1:{server.server_port}/{secret}/"
        s = session.state()
        print(f"Recorded so far: {s['minutes']:.1f} minutes. Sentences left: {s['left']}.")
        print("\nOpen this on your phone (or scan the code):\n")
        show_qr(url)
        print(f"  {url}\n")
        print("Recordings are saved to", ds.root)
        print("Keep this window open while you record. Press Ctrl+C here when you're done.")
        threading.Event().wait()
    except KeyboardInterrupt:
        print(f"\nDone. Total: {ds.seconds() / 60:.1f} minutes. When you have 10+, run: better-voice-train")
    finally:
        server.shutdown()
        if proc:
            proc.terminate()
