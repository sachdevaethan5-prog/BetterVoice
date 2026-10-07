import json
import tempfile
import unittest
from pathlib import Path

from better_voice.cleanup import clean
from better_voice.config import Config


class CleanupTest(unittest.TestCase):
    def test_removes_fillers(self):
        self.assertEqual(clean("um so I think, uh, we should go"), "So I think, we should go")

    def test_keeps_words_containing_filler_letters(self):
        self.assertEqual(clean("the umbrella is huge"), "The umbrella is huge")

    def test_collapses_stutters(self):
        self.assertEqual(clean("I I think the the plan works."), "I think the plan works.")
        self.assertEqual(clean("She had had enough."), "She had had enough.")

    def test_capitalizes_sentences_and_i(self):
        self.assertEqual(clean("hello there. how are you? i am fine"), "Hello there. How are you? I am fine")

    def test_new_line_and_paragraph(self):
        self.assertEqual(clean("first item new line second item"), "First item\nSecond item")
        self.assertEqual(clean("Hi Sam. New paragraph. Thanks for the notes."), "Hi Sam.\n\nThanks for the notes.")

    def test_scratch_that(self):
        self.assertEqual(clean("Let's meet at five. Scratch that. Let's meet at six."), "Let's meet at six.")
        self.assertEqual(clean("Buy milk. Call mom, scratch that, call dad."), "Buy milk. Call dad.")

    def test_replacements_and_dictionary(self):
        out = clean(
            "send it to my email and loop in kubernetes",
            dictionary=["Kubernetes"],
            replacements={"my email": "me@example.com"},
        )
        self.assertEqual(out, "Send it to me@example.com and loop in Kubernetes")

    def test_empty(self):
        self.assertEqual(clean("   "), "")
        self.assertEqual(clean("um."), "")


class ConfigTest(unittest.TestCase):
    def test_creates_default_then_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            cfg = Config.load(path)
            self.assertTrue(path.exists())
            cfg.dictionary = ["Sam"]
            cfg.save(path)
            self.assertEqual(Config.load(path).dictionary, ["Sam"])

    def test_ignores_unknown_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text(json.dumps({"model": "small.en", "bogus": 1}))
            self.assertEqual(Config.load(path).model, "small.en")


if __name__ == "__main__":
    unittest.main()
