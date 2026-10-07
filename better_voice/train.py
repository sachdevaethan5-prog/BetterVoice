"""Train your own speech model on your voice recordings.

Starts from the open Moonshine model (MIT license) and fine-tunes it on the clips in your
data folder, so it learns your voice, accent and words. About 1 clip in 10 is held back and
never trained on; the model is scored on those before and after, so you see whether it
really got better for you. The best version is saved as your model.

    better-voice-train                 # train, then point better-voice at the result
    better-voice-train --epochs 15     # more passes over the data
    better-voice-train --base UsefulSensors/moonshine-tiny   # smaller, faster model

A GPU makes this take minutes instead of hours. Free option: Google Colab (GPU runtime),
upload the data folder, pip install this project, run the same command.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import zlib
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, Transcriber, encrypt_model_dir, load_audio, prepare, save_private
from .config import Config, data_dir
from .dataset import Clip, Dataset, private
from .wer import wer


MIN_MINUTES = 10  # less than this is too little to learn a voice without overfitting
GENTLE_UNDER_MINUTES = 20  # below this, only the text half of the model is trained
GENERAL_SLACK = 0.02  # allowed loss of accuracy on other people's speech (2 points)
BASE = "UsefulSensors/moonshine-base"
GENERAL = Path(__file__).parent / "data" / "general"


def split(clips: list[Clip], eval_frac: float, min_eval: int = 15) -> tuple[list[Clip], list[Clip]]:
    """The same clips are held back every run (by file name), so scores stay comparable."""
    ranked = sorted(clips, key=lambda c: zlib.crc32(c.file.encode()))
    n_eval = min(max(min_eval, round(len(clips) * eval_frac)), len(clips) // 4) if len(clips) >= 5 else 0
    return ranked[n_eval:], ranked[:n_eval]


def general_check() -> tuple[list[Clip], dict[str, np.ndarray]]:
    """Other people reading (shipped with better-voice), to catch a model that forgot them."""
    import csv

    with (GENERAL / "metadata.csv").open(newline="", encoding="utf-8") as f:
        clips = [Clip(r["file"], r["text"], "general", True) for r in csv.DictReader(f)]
    return clips, {c.file: prepare(load_audio(GENERAL / c.file)) for c in clips}


def accepted(base_wer: float, new_wer: float, base_general: float, new_general: float) -> tuple[bool, str]:
    """A new model is used only if it's better on your voice and not worse on everyone else's."""
    if new_wer >= base_wer:
        return False, "it wasn't better on your held-back recordings"
    if new_general > base_general + GENERAL_SLACK:
        return False, (f"it got worse on general speech ({base_general:.1%} -> {new_general:.1%} wrong), "
                       "which would make everyday dictation worse")
    return True, ""


def augment(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small changes in loudness and background hiss, so a few minutes of data go further."""
    out = audio * rng.uniform(0.6, 1.4)
    if rng.random() < 0.5:
        out = out + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.001, 0.008), len(out))
    return np.clip(out, -1, 1).astype(np.float32)


class Trainer(Transcriber):
    """The app's Transcriber, plus learning from clips and saving the result."""

    def __init__(self, base: str, device: str) -> None:
        super().__init__(base, device)
        self.torch = self._torch
        self.eos = self.model.config.eos_token_id

    def batch(self, audios: list[np.ndarray], texts: list[str]):
        torch = self.torch
        feats = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, return_attention_mask=True)
        # The tokenizer starts with <s>, which is also the decoder's start token; the model adds that
        # itself when it shifts the labels, so labels are the words plus the end token.
        ids = [self.processor.tokenizer(t).input_ids[1:] + [self.eos] for t in texts]
        width = max(map(len, ids))
        labels = torch.full((len(ids), width), -100, dtype=torch.long)
        for i, seq in enumerate(ids):
            labels[i, : len(seq)] = torch.tensor(seq)
        return {k: v.to(self.device) for k, v in feats.items()}, labels.to(self.device)

    def loss(self, feats, labels):
        """Cross-entropy of the next word, computed here rather than by the model: some
        transformers versions (4.57) shift the labels a second time inside the model, which
        silently trains it to predict the word after the right one."""
        torch = self.torch
        start = self.model.config.decoder_start_token_id
        pad = self.model.config.pad_token_id
        dec_in = torch.cat([torch.full_like(labels[:, :1], start), labels[:, :-1]], dim=1)
        dec_in = dec_in.masked_fill(dec_in == -100, pad)
        logits = self.model(**feats, decoder_input_ids=dec_in).logits
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

    def score(self, clips: list[Clip], audio: dict[str, np.ndarray]) -> tuple[float, list[str]]:
        # The clips are already prepared, so they go straight to the model (no second trim).
        self.model.eval()
        hyps = [self._one(audio[c.file]) for c in clips]
        return wer([c.text for c in clips], hyps), hyps

    def save(self, out: Path, encrypted: bool) -> None:
        if encrypted:
            save_private(self.model, self.processor, out)
        else:  # training on a plain pack elsewhere; encrypt it at home with --import-model
            out.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(out)
            self.processor.save_pretrained(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="better-voice-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=None, help="data folder (default: your user data folder)")
    p.add_argument("--base", default="UsefulSensors/moonshine-base", help="model to start from")
    p.add_argument("--out", type=Path, default=None, help="where to save your model (default: <data>/voice-model)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3, help="stop after this many passes without improvement")
    p.add_argument("--encoder", choices=["auto", "freeze", "train"], default="auto",
                   help=f"train the audio half too? auto = only with {GENTLE_UNDER_MINUTES}+ minutes of recordings")
    p.add_argument("--force", action="store_true", help=f"train even with under {MIN_MINUTES} minutes of recordings")
    p.add_argument("--reset", action="store_true", help="switch better-voice back to the stock model and stop")
    p.add_argument("--import-model", type=Path, metavar="FOLDER",
                   help="bring home a model trained elsewhere (e.g. Colab): encrypt it and use it")
    p.add_argument("--include-unchecked", action="store_true", help="also train on dictations you haven't corrected")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-apply", action="store_true", help="don't switch better-voice to the new model")
    args = p.parse_args()

    if args.reset:
        cfg = Config.load()
        cfg.model = BASE
        cfg.save()
        raise SystemExit(f"better-voice is back on the stock model ({BASE}). Your recordings are untouched.")

    cfg = Config.load()
    if args.import_model:
        if not (args.import_model / "model.safetensors").exists():
            raise SystemExit(f"No model.safetensors in {args.import_model}.")
        out = args.out or data_dir() / "voice-model"
        if args.import_model.resolve() != out.resolve():
            import shutil

            shutil.copytree(args.import_model, out, dirs_exist_ok=True)
        if cfg.encrypt:
            encrypt_model_dir(out)
        cfg.model = str(out)
        cfg.save()
        raise SystemExit(f"Imported your model into {out}{' (encrypted)' if cfg.encrypt else ''}. Delete the "
                         f"plain copy at {args.import_model} (and anywhere you uploaded it).")

    ds = Dataset(args.data) if args.data else private(data_dir(), cfg.encrypt)
    out = args.out or ds.root / "voice-model"
    clips = ds.rows(checked_only=not args.include_unchecked)
    train_set, eval_set = split(clips, args.eval_frac)
    if len(clips) < 30 or not eval_set:
        raise SystemExit(f"Only {len(clips)} usable recordings in {ds.root}. Record more with better-voice-record first.")
    have = ds.seconds(checked_only=not args.include_unchecked) / 60
    if have < MIN_MINUTES and not args.force:
        raise SystemExit(f"You have {have:.1f} minutes of recordings; training needs at least {MIN_MINUTES} "
                         "(about 150 sentences) or it tends to make dictation worse. Keep going with better-voice-record.")
    freeze = args.encoder == "freeze" or (args.encoder == "auto" and have < GENTLE_UNDER_MINUTES)

    rng = random.Random(args.seed)
    trainer = Trainer(args.base, args.device)
    torch = trainer.torch
    torch.manual_seed(args.seed)
    audio = {c.file: prepare(ds.load(c)) for c in clips}
    minutes = sum(len(audio[c.file]) for c in train_set) / SAMPLE_RATE / 60
    print(f"Training on {len(train_set)} clips ({minutes:.1f} min), testing on {len(eval_set)} held-back clips, device {trainer.device}.")

    base_wer, _ = trainer.score(eval_set, audio)
    general, general_audio = general_check()
    base_general, _ = trainer.score(general, general_audio)
    print(f"Starting model: {base_wer:.1%} of words wrong on your held-back clips, {base_general:.1%} on general speech.")
    if freeze:
        print(f"Training only the text half of the model (gentler; the audio half joins at {GENTLE_UNDER_MINUTES}+ minutes).")

    for n, prm in trainer.model.named_parameters():
        prm.requires_grad = not (freeze and ".encoder." in f".{n}")
    params = [prm for prm in trainer.model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train_set) / args.batch)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))

    best_wer, best_epoch, best_general, stale = base_wer, 0, base_general, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        trainer.model.train()
        order = train_set[:]
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), args.batch):
            chunk = order[i : i + args.batch]
            feats, labels = trainer.batch([augment(audio[c.file], rng) for c in chunk], [c.text for c in chunk])
            loss = trainer.loss(feats, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        score, _ = trainer.score(eval_set, audio)
        general_score, _ = trainer.score(general, general_audio)
        ok, _why = accepted(best_wer, score, base_general, general_score)
        mark = ""
        if ok:
            best_wer, best_epoch, best_general, stale = score, epoch, general_score, 0
            trainer.save(out, encrypted=ds.encrypted)
            mark = "  saved"
        else:
            stale += 1
        print(f"Pass {epoch}: training loss {np.mean(losses):.3f}, {score:.1%} words wrong on yours, "
              f"{general_score:.1%} on general speech{mark}")
        if stale >= args.patience:
            print("No improvement for a while, stopping early.")
            break

    report = {
        "base_model": args.base,
        "train_clips": len(train_set),
        "train_minutes": round(minutes, 1),
        "eval_clips": len(eval_set),
        "base_wer": round(base_wer, 4),
        "your_model_wer": round(best_wer, 4),
        "base_general_wer": round(base_general, 4),
        "your_model_general_wer": round(best_general, 4),
        "encoder_trained": not freeze,
        "best_pass": best_epoch,
        "minutes_taken": round((time.time() - t0) / 60, 1),
    }
    print(f"\nYour recordings: {base_wer:.1%} wrong before, {best_wer:.1%} with your model.")
    print(f"General speech: {base_general:.1%} wrong before, {best_general:.1%} with your model.")
    if best_epoch == 0:
        print("No training pass was better on your voice without getting worse on general speech, so nothing "
              "was changed. Record more, then train again.")
        return
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved your model to {out} (details in report.json).")
    if not args.no_apply:
        cfg = Config.load()
        cfg.model = str(out)
        cfg.save()
        print("better-voice will use it from its next start.")


if __name__ == "__main__":
    main()
