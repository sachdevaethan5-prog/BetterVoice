"""better-voice: speech-to-text with your own model, for testing and for your own apps.

Dictating into other apps is the Better Voice desktop app's job (app/).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from . import cleanup
from .audio import SAMPLE_RATE, Transcriber, load_audio
from .config import Config, config_dir, data_dir


def process(raw: str, cfg: Config) -> str:
    text = raw
    if cfg.cleanup:
        text = cleanup.clean(text, cfg.dictionary, cfg.replacements)
    if cfg.llm_cleanup and text:
        text = cleanup.llm_polish(text, cfg.ollama_url, cfg.ollama_model)
    return text


def log_history(raw: str, text: str, seconds: float) -> None:
    path = config_dir() / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "seconds": round(seconds, 1),
             "raw": raw, "text": text}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_model(cfg: Config) -> Transcriber:
    print(f"[better-voice] loading speech model '{cfg.model}' (first run downloads it)...")
    return Transcriber(cfg.model, cfg.device)


def keep(audio, raw: str, cfg: Config) -> None:
    """Save a dictation as unchecked training data (see better_voice/dataset.py)."""
    if cfg.keep_audio and raw:
        from .dataset import private

        private(data_dir(), cfg.encrypt).add(audio, raw, "dictation", checked=False)


def transcribe_file(path: Path, cfg: Config) -> None:
    """Transcribe and clean an audio file; handy for testing settings without a mic."""
    print(process(load_model(cfg).transcribe(load_audio(path)), cfg))


def run_server(cfg: Config, host: str, port: int) -> None:
    from .serve import ensure_token, serve

    token = ensure_token(cfg)
    transcriber = load_model(cfg)

    def transcribe(fileobj) -> dict:
        audio = load_audio(fileobj)
        raw = transcriber.transcribe(audio)
        text = process(raw, cfg)
        keep(audio, raw, cfg)
        if cfg.history and raw:
            log_history(raw, text, len(audio) / SAMPLE_RATE)
        return {"text": text, "raw": raw, "seconds": round(len(audio) / SAMPLE_RATE, 2)}

    serve(transcribe, token, host, port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="better-voice", description=__doc__)
    parser.add_argument("--config", type=Path, help="path to a config.json (default: your user config folder)")
    parser.add_argument("--model", help="speech model: a Hugging Face name or the folder of your trained model")
    parser.add_argument("--llm", action="store_true", help="also polish text with a local Ollama model")
    parser.add_argument("--file", type=Path, help="transcribe this audio file and print the result")
    parser.add_argument("--clean", metavar="TEXT", help="run the text cleanup on TEXT and print it")
    parser.add_argument("--serve", action="store_true", help="run as a speech-to-text server for other apps")
    parser.add_argument("--host", default="127.0.0.1", help="server address (with --serve)")
    parser.add_argument("--port", type=int, default=8765, help="server port (with --serve)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.model:
        cfg.model = args.model
    if args.llm:
        cfg.llm_cleanup = True

    if args.clean is not None:
        print(process(args.clean, cfg))
    elif args.serve:
        run_server(cfg, args.host, args.port)
    elif args.file:
        transcribe_file(args.file, cfg)
    else:
        parser.print_help()
        print("\nTo dictate into any app, use the Better Voice desktop app (see app/README.md).")


if __name__ == "__main__":
    main()
