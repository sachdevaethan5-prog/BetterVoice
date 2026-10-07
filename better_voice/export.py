"""Convert your trained model into the files the Better Voice desktop app loads.

    better-voice-export                        # your trained model -> the app's models folder
    better-voice-export --model <folder> --out <folder>

Your trained model stays encrypted; this decrypts it in memory to convert it. The app's copy
is a normal file in the app's own folder, by design: the app loads it like any other model. The app runs Moonshine through ONNX Runtime and expects one folder holding
encoder_model.onnx, decoder_model_merged.onnx and tokenizer.json, compressed to 8-bit like
the stock model it ships with. Needs the export extras: pip install -e ".[export]"
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from .config import Config, data_dir, user_dir

FILES = ("encoder_model.onnx", "decoder_model_merged.onnx", "tokenizer.json")
FOLDER_NAME = "my-voice-moonshine"


def app_models_dir() -> Path:
    """Where the desktop app looks for models (its app data folder + /models)."""
    return user_dir(data=True) / "com.bettervoice.app" / "models"


def export(model_dir: Path, out_dir: Path, quantize: bool = True) -> Path:
    """model_dir: a model folder (encrypted weights are decrypted in memory) or a Hugging Face name."""
    from optimum.exporters.onnx import onnx_export_from_model

    from .audio import load_moonshine

    processor, model = load_moonshine(str(model_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_export_from_model(model.eval(), output=tmp, task="automatic-speech-recognition-with-past")
        processor.tokenizer.save_pretrained(tmp)
        for name in FILES[:2]:
            src, dst = Path(tmp) / name, out_dir / name
            if quantize:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                # Same recipe as the stock model the app ships: only the matrix multiplies are
                # made 8-bit (8-bit audio convolutions wreck the output), including the ones
                # inside the merged decoder's cache branches.
                quantize_dynamic(
                    str(src),
                    str(dst),
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["MatMul"],
                    extra_options={"EnableSubgraph": True},
                )
            else:
                shutil.copy(src, dst)
        shutil.copy(Path(tmp) / "tokenizer.json", out_dir / "tokenizer.json")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(prog="better-voice-export", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=None, help="trained model folder (default: <data>/voice-model)")
    p.add_argument("--out", type=Path, default=None, help=f"output folder (default: the app's models folder/{FOLDER_NAME})")
    p.add_argument("--no-quantize", action="store_true", help="keep full precision (about 4x bigger, slightly more accurate)")
    args = p.parse_args()

    model = args.model or data_dir() / "voice-model"
    if not (model / "config.json").exists():
        cfg = Config.load()
        hint = "" if cfg.model.startswith("UsefulSensors/") else f" (better-voice is set to {cfg.model})"
        raise SystemExit(f"No trained model at {model}{hint}. Run better-voice-train first, or pass --model.")
    import json

    size = json.loads((model / "config.json").read_text()).get("hidden_size")
    if size != 416:
        raise SystemExit(f"The app runs Moonshine Base, but this model is a different size (hidden_size {size}). "
                         "Train from UsefulSensors/moonshine-base (the default).")
    out = args.out or app_models_dir() / FOLDER_NAME
    print(f"Converting {model} -> {out} ...")
    export(model, out, quantize=not args.no_quantize)
    size = sum((out / f).stat().st_size for f in FILES) / 1e6
    print(f"Done: {size:.0f} MB. In Better Voice, open Settings > Models and pick \"My voice\" "
          "(press Rescan if the app was already open).")


if __name__ == "__main__":
    main()
