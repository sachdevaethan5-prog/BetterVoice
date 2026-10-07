"""Record yourself reading sentences: the training data for your own speech model.

    better-voice-record              # go through the prompts you haven't recorded yet
    better-voice-record --minutes 10 # stop after 10 more minutes of audio
    better-voice-record --phone      # record with your phone's mic instead
    better-voice-record --review     # fix the text of saved dictations so training can use them
    better-voice-record --pack DIR   # plain copy for training on another computer (e.g. Colab)
    better-voice-record --show-key   # the encryption key, to save in a password manager

Recordings are encrypted on this computer. --phone and --pack send voice data off it, so
they ask you to accept the sharing agreement first.

For each sentence: press Enter, read it out loud, press Enter again. Then Enter to keep it,
r to redo, s to skip, or q to quit. Recordings are kept between sessions, so do a few
minutes whenever you have time. About 30 minutes in total is a good first goal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, Recorder
from .config import Config, data_dir
from . import consent, vault
from .dataset import Dataset, private
from .vault import VaultError
from .prompts import build


def main() -> None:
    parser = argparse.ArgumentParser(prog="better-voice-record", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="data folder (default: your user data folder)")
    parser.add_argument("--minutes", type=float, default=None, help="stop after this many minutes of new audio")
    parser.add_argument("--phone", action="store_true", help="record with your phone: prints a link to open on it")
    parser.add_argument("--local", "--no-tunnel", dest="local", action="store_true",
                        help="the recording page, on this computer only (used by the app's Train my voice page)")
    parser.add_argument("--status", action="store_true", help="recordings and model status as JSON (used by the app)")
    parser.add_argument("--review", action="store_true", help="check and fix saved dictations")
    parser.add_argument("--pack", type=Path, metavar="DIR", help="plain copy of your recordings for training elsewhere")
    parser.add_argument("--show-key", action="store_true", help="print the encryption key (save it in a password manager)")
    args = parser.parse_args()

    cfg = Config.load()
    try:
        ds = Dataset(args.data) if args.data else private(data_dir(), cfg.encrypt)
    except VaultError as exc:
        raise SystemExit(str(exc))
    if args.status:
        return print(json.dumps(status(ds, cfg)))
    if args.show_key:
        print("Your voice data key. Anyone with it and your files can read your recordings, so keep it\n"
              "only in a password manager. To restore it on a new computer, set BETTER_VOICE_KEY to it.\n")
        return print(vault.key().decode())
    if args.review:
        return review(ds)
    if args.pack:
        what = f"A plain, unencrypted copy of all your recordings and transcripts will be written to:\n  {args.pack}"
        if consent.require(cfg, what):
            out = ds.export_plain(args.pack)
            print(f"Packed {len(out.rows(checked_only=False))} recordings into {args.pack}. Train there with "
                  f"better-voice-train --data <that folder>, bring the voice-model folder back with "
                  f"better-voice-train --import-model <folder>, then delete every plain copy.")
        return
    if args.phone or args.local:
        from .phone import run

        tunnel = not args.local
        what = ("Recording with your phone sends your audio from the phone, through Cloudflare's tunnel\n"
                "(trycloudflare.com), to this computer.")
        if tunnel and not consent.require(cfg, what):
            return
        try:
            return run(ds, cfg.dictionary, port=8766 if tunnel else 0, tunnel=tunnel)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(str(exc))
    done = {c.text.lower() for c in ds.rows(checked_only=False)}
    todo = [p for p in build(ds.root, cfg.dictionary) if p.lower() not in done]
    total = ds.seconds()
    print(f"Data folder: {ds.root}")
    print(f"Recorded so far: {total / 60:.1f} minutes. Sentences left: {len(todo)}.")
    print("Tip: speak the way you normally talk, in the room you usually use.\n")

    rec = Recorder()
    new_seconds = 0.0
    for i, sentence in enumerate(todo, 1):
        if args.minutes is not None and new_seconds >= args.minutes * 60:
            break
        while True:
            print(f"[{i}/{len(todo)}]  {sentence}")
            if input("  Enter to start (q to quit) ").strip().lower() == "q":
                return finish(ds, new_seconds)
            rec.start()
            input("  Recording... Enter when done ")
            audio = rec.stop()
            secs = len(audio) / SAMPLE_RATE
            peak = float(np.abs(audio).max()) if len(audio) else 0.0
            if secs < 0.5:
                print(f"  That was only {secs:.1f}s. Press Enter, read the whole sentence, then press Enter.")
                continue
            if peak < 0.003:
                print(f"  The mic picked up nothing (level {peak:.3f}). Check the input device in Windows sound "
                      "settings and that the mic isn't muted, then try again.")
                continue
            if peak < 0.03:
                # Quiet but usable: training boosts it to a steady level, like the app does.
                print(f"  (Quiet recording, level {peak:.3f}. It's kept, but turning up the mic helps.)")
            if peak > 0.99:
                print("  (That clipped a little; move back from the mic if it keeps happening.)")
            choice = input(f"  {secs:.1f}s. Enter keep, r redo, s skip, q quit: ").strip().lower()
            if choice == "r":
                continue
            if choice in ("", "q"):
                # Trim the key-press noise at both ends; keep a little room around the speech.
                cut = int(0.1 * SAMPLE_RATE)
                ds.add(audio[cut:-cut] if len(audio) > 4 * cut else audio, sentence, "prompt", True)
                new_seconds += secs
            if choice == "q":
                return finish(ds, new_seconds)
            break
    finish(ds, new_seconds)


def status(ds: Dataset, cfg: Config) -> dict:
    from .train import MIN_MINUTES

    rows = ds.rows(checked_only=False)
    done = {c.text.lower() for c in rows}
    return {
        "minutes": round(ds.seconds() / 60, 2),
        "recordings": len(rows),
        "sentences_left": sum(1 for s in build(ds.root, cfg.dictionary) if s.lower() not in done),
        "min_minutes": MIN_MINUTES,
        "trained": (ds.root / "voice-model" / "config.json").exists(),
        "using_trained_model": not cfg.model.startswith("UsefulSensors/"),
        "encrypted": ds.encrypted,
        "data_folder": str(ds.root),
    }


def review(ds: Dataset) -> None:
    """Go through dictations saved by keep_audio: confirm or correct each transcript."""
    todo = [c for c in ds.rows(checked_only=False) if not c.checked]
    if not todo:
        return print("Nothing to review. Dictations are saved here when keep_audio is on.")
    print(f"{len(todo)} dictations to review. For each: Enter if the text is exactly what you said,\n"
          "type the correct text, d to delete the recording, or q to stop.\n")
    for i, c in enumerate(todo, 1):
        answer = input(f"[{i}/{len(todo)}] {c.text}\n  > ").strip()
        if answer.lower() == "q":
            break
        if answer.lower() == "d":
            ds.remove(c.file)
        else:
            ds.update(c.file, answer or c.text, checked=True)
    print("Saved. The next training run uses the checked ones.")


def finish(ds: Dataset, new_seconds: float) -> None:
    print(f"\nSaved {new_seconds / 60:.1f} new minutes. Total: {ds.seconds() / 60:.1f} minutes in {ds.root}")
    print("When you have about 30 minutes, run: better-voice-train")


if __name__ == "__main__":
    main()
