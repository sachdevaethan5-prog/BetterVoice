"""Tests for the training data, scoring, phone recorder and server. No model or mic needed."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import os

import numpy as np

try:  # every test here uses a throwaway key; real use keeps it in the OS credential store
    from cryptography.fernet import Fernet

    os.environ.setdefault("BETTER_VOICE_KEY", Fernet.generate_key().decode())
except ImportError:
    pass

from better_voice.config import Config
from better_voice.dataset import Dataset
from better_voice.prompts import build
from better_voice.serve import make_handler
from better_voice.train import split
from better_voice.wer import normalize, wer


class WerTest(unittest.TestCase):
    def test_ignores_formatting(self):
        self.assertEqual(wer(["Meet me at 4:30, okay?"], ["meet me at 4:30 okay"]), 0.0)
        self.assertEqual(normalize("It's $23.50."), ["it's", "$23.50"])

    def test_counts_errors(self):
        self.assertAlmostEqual(wer(["call the bank today"], ["call a bank"]), 0.5)  # 1 swap + 1 drop of 4


class DatasetTest(unittest.TestCase):
    def test_add_and_filter(self):
        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")
        with tempfile.TemporaryDirectory() as d:
            ds = Dataset(Path(d))
            tone = np.sin(np.linspace(0, 300, 16000)).astype(np.float32) * 0.3
            ds.add(tone, "What's due tomorrow?", "prompt", True)
            ds.add(tone, "model guess", "dictation", False)
            self.assertEqual([c.text for c in ds.rows()], ["What's due tomorrow?"])
            self.assertEqual(len(ds.rows(checked_only=False)), 2)
            self.assertAlmostEqual(ds.seconds(), 1.0, places=2)


class PhoneSessionTest(unittest.TestCase):
    def test_record_skip_and_redo(self):
        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")
        from better_voice.phone import Session

        with tempfile.TemporaryDirectory() as d:
            s = Session(Dataset(Path(d)), [])
            first = s.state()["sentence"]
            speech = (np.sin(np.linspace(0, 2000, 48000)) * 0.3).astype(np.float32)
            saved = s.save(speech, first)
            self.assertNotEqual(s.state()["sentence"], first)
            with self.assertRaises(ValueError):
                s.save(np.zeros(48000, dtype=np.float32), "Silent sentence.")
            second = s.state()["sentence"]
            s.skipped.add(second.lower())
            self.assertNotIn(s.state()["sentence"], (first, second))
            s.undo(saved["file"])
            self.assertEqual(s.state()["sentence"], first)  # redo brings it back
            self.assertEqual(Dataset(Path(d)).rows(), [])


def needs(*mods):
    for m in mods:
        try:
            __import__(m)
        except ImportError:
            return unittest.skip(f"{m} not installed")
    return lambda f: f


SPEECH = (np.sin(np.linspace(0, 2000, 32000)) * 0.3).astype(np.float32)


class PrivacyTest(unittest.TestCase):
    @needs("soundfile", "cryptography")
    def test_recordings_are_encrypted_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Dataset(Path(d))
            clip = ds.add(SPEECH, "Call the dentist tomorrow.", "prompt", True)
            files = [f.read_bytes() for f in Path(d).rglob("*") if f.is_file()]
            # Every file is a Fernet token (random-looking base64, so look at the start, not
            # for words that can turn up by chance inside it).
            self.assertTrue(files and all(b.startswith(b"gAAAAA") for b in files))
            self.assertFalse(list(Path(d).rglob("*.csv")))
            self.assertEqual(ds.rows()[0].text, "Call the dentist tomorrow.")
            self.assertEqual(len(ds.load(clip)), len(SPEECH))
            self.assertAlmostEqual(ds.seconds(), 2.0)

    @needs("soundfile", "cryptography")
    def test_old_plain_recordings_get_encrypted(self):
        from better_voice.dataset import private

        with tempfile.TemporaryDirectory() as d:
            Dataset(Path(d), encrypted=False).add(SPEECH, "Pay the phone bill.", "prompt", True)
            ds = private(Path(d))
            self.assertTrue(ds.encrypted)
            self.assertFalse(list(Path(d).rglob("*.wav")) + list(Path(d).rglob("*.csv")))
            self.assertEqual([c.text for c in ds.rows()], ["Pay the phone bill."])
            self.assertEqual(len(ds.load(ds.rows()[0])), len(SPEECH))

    @needs("soundfile", "cryptography")
    def test_encryption_can_be_switched_off_and_on(self):
        from better_voice.dataset import private

        with tempfile.TemporaryDirectory() as d:
            private(Path(d)).add(SPEECH, "Move my study session.", "prompt", True)
            off = private(Path(d), encrypt=False)
            self.assertFalse(off.encrypted)
            self.assertEqual(len(list(Path(d).rglob("*.wav"))), 1)
            self.assertFalse(list(Path(d).rglob("*.enc")))
            on = private(Path(d), encrypt=True)
            self.assertFalse(list(Path(d).rglob("*.wav")) + list(Path(d).rglob("*.csv")))
            self.assertEqual([c.text for c in on.rows()], ["Move my study session."])
            self.assertEqual(len(on.load(on.rows()[0])), len(SPEECH))

    @needs("soundfile", "cryptography")
    def test_pack_is_plain_and_complete(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Dataset(Path(d) / "mine")
            ds.add(SPEECH, "Read me my tasks.", "prompt", True)
            ds.add(SPEECH, "guess", "dictation", False)
            ds.update(ds.rows(checked_only=False)[1].file, "Guessed right.", checked=True)
            pack = ds.export_plain(Path(d) / "pack")
            self.assertFalse(pack.encrypted)
            self.assertEqual(sorted(c.text for c in Dataset(Path(d) / "pack").rows()), ["Guessed right.", "Read me my tasks."])
            self.assertEqual(len(list((Path(d) / "pack" / "clips").glob("*.wav"))), 2)

    @needs("cryptography")
    def test_wrong_key_is_a_clear_error(self):
        from cryptography.fernet import Fernet

        from better_voice import vault

        locked = Fernet(Fernet.generate_key()).encrypt(b"secret")
        with self.assertRaises(vault.VaultError):
            vault.decrypt(locked)

    def test_agreement_is_required_and_remembered(self):
        from better_voice import consent

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            cfg = Config.load(path)
            cfg.save = lambda: Config.save(cfg, path)
            self.assertFalse(consent.require(cfg, "test", ask=lambda _: ""))
            self.assertFalse(consent.require(cfg, "test", ask=lambda _: "ok"))
            self.assertEqual(Config.load(path).sharing_agreement, {})
            self.assertTrue(consent.require(cfg, "test", ask=lambda _: "i agree"))
            self.assertEqual(Config.load(path).sharing_agreement["version"], consent.VERSION)
            self.assertTrue(consent.require(Config.load(path), "test", ask=lambda _: self.fail("asked twice")))

    @needs("torch", "transformers", "safetensors", "cryptography")
    def test_trained_model_weights_are_encrypted(self):
        import torch
        from transformers import AutoTokenizer, MoonshineConfig, MoonshineForConditionalGeneration

        from better_voice.audio import WEIGHTS_ENC, load_moonshine, save_private

        cfg = MoonshineConfig(vocab_size=64, hidden_size=32, intermediate_size=64, encoder_num_hidden_layers=1,
                              decoder_num_hidden_layers=1, encoder_num_attention_heads=2, decoder_num_attention_heads=2)
        model = MoonshineForConditionalGeneration(cfg)

        class Proc:  # stands in for the processor; only saving is used
            def save_pretrained(self, out):
                (Path(out) / "preprocessor_config.json").write_text("{}")

        with tempfile.TemporaryDirectory() as d:
            save_private(model, Proc(), Path(d))
            self.assertTrue((Path(d) / WEIGHTS_ENC).exists())
            self.assertFalse((Path(d) / "model.safetensors").exists())
            from safetensors.torch import load

            from better_voice import vault

            back = load(vault.decrypt((Path(d) / WEIGHTS_ENC).read_bytes()))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, back[k]), k)


class SplitTest(unittest.TestCase):
    def test_same_clips_held_back_every_run(self):
        from better_voice.dataset import Clip

        clips = [Clip(f"clips/{i}.wav", "x", "prompt", True) for i in range(50)]
        train, held = split(clips, 0.1)
        self.assertEqual(len(held), 12)  # at least 15 wanted, capped at a quarter of 50
        self.assertEqual(len(train) + len(held), 50)
        self.assertEqual(split(list(reversed(clips)), 0.1)[1], held)
        self.assertEqual(len(split(clips * 10, 0.1)[1]), 50)  # 10% once there are plenty

    def test_only_better_models_that_still_hear_others_are_used(self):
        from better_voice.train import accepted

        self.assertTrue(accepted(0.20, 0.12, 0.10, 0.11)[0])
        self.assertFalse(accepted(0.20, 0.20, 0.10, 0.10)[0])  # not better on your voice
        ok, why = accepted(0.20, 0.05, 0.10, 0.25)  # learned you, forgot everyone else
        self.assertFalse(ok)
        self.assertIn("general speech", why)


class PromptsTest(unittest.TestCase):
    def test_large_sentence_bank(self):
        with tempfile.TemporaryDirectory() as d:
            out = build(Path(d), [])
        self.assertGreater(len(out), 2000)
        self.assertTrue(all(s.strip() and not s.startswith("#") for s in out))

    def test_own_prompts_first_and_no_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "prompts.txt").write_text("# my sentences\nMy own sentence.\nWhat's due tomorrow?\n")
            out = build(Path(d), ["Seattle"])
            self.assertEqual(out[0], "My own sentence.")
            self.assertEqual(len(out), len({s.lower() for s in out}))
            self.assertTrue(any("Seattle" in s for s in out))


class ExportTest(unittest.TestCase):
    def run_export(self, config: dict):
        import sys
        from unittest import mock

        from better_voice import export

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps(config))
            with mock.patch.object(sys, "argv", ["better-voice-export", "--model", d, "--out", d + "/out"]), \
                    mock.patch.object(export, "export") as fake:
                try:
                    export.main()
                except SystemExit as e:
                    return str(e), fake.called
            return "", fake.called

    def test_rejects_models_the_app_cant_run(self):
        msg, ran = self.run_export({"hidden_size": 288})  # Moonshine Tiny
        self.assertIn("Moonshine Base", msg)
        self.assertFalse(ran)


class ServerTest(unittest.TestCase):
    def setUp(self):
        def fake(fileobj):
            data = fileobj.read()
            if data != b"RIFFaudio":
                raise ValueError("Format not recognised")
            return {"text": "Hello.", "raw": "hello", "seconds": 1.0}

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake, "secret"))
        self.server.RequestHandlerClass.log_message = lambda *a: None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()

    def post(self, body, token="secret"):
        req = urllib.request.Request(self.url + "/transcribe", data=body, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_needs_token(self):
        self.assertEqual(self.post(b"RIFFaudio", token="wrong")[0], 401)

    def test_transcribes(self):
        self.assertEqual(self.post(b"RIFFaudio"), (200, {"text": "Hello.", "raw": "hello", "seconds": 1.0}))

    def test_bad_audio(self):
        self.assertEqual(self.post(b"nope")[0], 400)

    def test_health(self):
        with urllib.request.urlopen(self.url + "/health") as r:
            self.assertEqual(json.loads(r.read()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
