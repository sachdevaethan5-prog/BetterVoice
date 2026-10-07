"""Speech-to-text server, so other apps (a voice assistant, a web page) can use your model.

    better-voice --serve                # listens on 127.0.0.1:8765
    better-voice --serve --port 9000 --host 0.0.0.0

POST /transcribe with a WAV/FLAC/OGG body and the header `Authorization: Bearer <token>`.
Reply: {"text": "<cleaned text>", "raw": "<model output>", "seconds": 2.4}.
GET /health needs no token. The token is created on first start and saved in config.json.
To reach it from the internet (Cloudflare), put it behind a Cloudflare Tunnel; never open the
port directly.
"""

from __future__ import annotations

import hmac
import io
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 10_000_000  # about 5 minutes of 16 kHz WAV


class Handler(BaseHTTPRequestHandler):
    """Shared by this server and the phone recorder: JSON and plain replies, never cached."""

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")  # the phone link carries a secret
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode())


def make_handler(transcribe, token: str):
    lock = threading.Lock()  # one transcription at a time; the model isn't thread-safe

    class Transcribe(Handler):
        def do_GET(self) -> None:
            if self.path == "/health":
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/transcribe":
                return self._json(404, {"error": "not found"})
            given = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(given.encode(), token.encode()):
                return self._json(401, {"error": "bad token"})
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_BYTES:
                return self._json(413 if size else 400, {"error": "send 1 byte to 10 MB of audio"})
            body = self.rfile.read(size)
            try:
                with lock:
                    result = transcribe(io.BytesIO(body))
            except Exception as exc:
                return self._json(400, {"error": f"couldn't read that audio: {exc}"})
            self._json(200, result)

        def log_message(self, fmt, *args) -> None:  # one short line per request
            print(f"[better-voice] {self.address_string()} {fmt % args}")

    return Transcribe


def ensure_token(cfg) -> str:
    if not cfg.server_token:
        cfg.server_token = secrets.token_urlsafe(32)
        cfg.save()
    return cfg.server_token


def serve(transcribe, token: str, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(transcribe, token))
    print(f"[better-voice] serving on http://{host}:{port}/transcribe (token in config.json)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[better-voice] bye")
