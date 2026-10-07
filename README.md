# Better Voice

Free voice dictation that can learn **your** voice.
Press a shortcut, talk, and clean text appears in whatever app you're typing in. It runs on
your computer: no account, no subscription, nothing sent to the cloud.

**[Download Better Voice](https://github.com/sachdevaethan5-prog/BetterVoice/releases/latest)**
(Windows, Mac with Apple Silicon, Linux)

## Get started

1. Download the installer for your computer from the link above: Windows `_x64-setup.exe`,
   Mac `.dmg`, Linux `.deb`.
2. Install it. The installers aren't signed yet, so your computer warns you the first time:
   on Windows click **More info → Run anyway**; on a Mac right-click the app and choose **Open**.
3. Open Better Voice, allow the microphone, and pick a speech model when it asks.
4. Click into any text box, press the shortcut shown in the app's settings, and talk. The text
   is typed in when you finish.

By installing or using Better Voice you accept the [Disclaimer](#disclaimer) below.

Out of the box it uses the open [Moonshine](https://github.com/usefulsensors/moonshine) model
(MIT license). To make it better at hearing you, train it on your own voice (below).

## How it's built

- **The desktop app** (`app/`): tray icon, settings, on-screen recording bubble, text cleanup,
  installers for Windows, macOS and Linux. Built on [Handy](https://github.com/cjpais/Handy)
  (MIT). This is what you dictate with.
- **The voice tools** (`better_voice/`, Python): record your voice, train your own copy of the
  model on it, convert it for the app, and serve it to your own apps.

You own the trained files. Your recordings and model stay on your computer, encrypted; see
[PRIVACY.md](PRIVACY.md).

## Install the voice tools

You need Python 3.10 or newer.

```bash
git clone https://github.com/sachdevaethan5-prog/BetterVoice.git better-voice
cd better-voice
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[export]"
```

## Train it on your voice

The easy way: open **Train voice** in the app. It shows how much you've recorded, opens the
recorder, trains and switches the app to your model, using the voice tools below (install
them first and enter their folder on that page). The same steps from a terminal:

**1. Record.** 10 minutes is the minimum to train, 30 is a good goal, a few minutes at a time.
There are about 2,200 sentences: everyday ones (assistant commands, texts, emails, school, money)
mixed with general ones from Mozilla Common Voice.

```bash
better-voice-record          # with this computer's mic: Enter to start, Enter to stop
better-voice-record --phone  # with your phone: open the link or scan the QR code
```

`--phone` sends your audio through Cloudflare, so it asks you to accept the sharing agreement
first ([PRIVACY.md](PRIVACY.md)). It needs the free `cloudflared` tool once (Windows: `winget install --id
Cloudflare.cloudflared`, Mac: `brew install cloudflared`), because phone browsers only allow the
microphone on https pages. Recordings save straight to this computer.

Put your own names and jargon in `dictionary` in the settings, and your own sentences (one per
line) in `prompts.txt` in the data folder. Those are read first.

Your recordings are encrypted as they're saved. Back up the key once with
`better-voice-record --show-key` and keep it in a password manager: without it, the recordings
can't be read on a new computer.

**2. Train.**

```bash
better-voice-train
```

It holds back some of your recordings and 20 clips of other people talking, scores the model on
both before and after, and only uses your model if it's better on your voice **and** not worse
on everyone else's. With under 20 minutes it trains only the gentler text half of the model.
`better-voice-train --reset` goes back to the stock model.

A GPU makes training take minutes instead of hours. Free option:
[Google Colab](https://colab.research.google.com) with a GPU runtime. That means your voice
leaves your computer, so `better-voice-record --pack <folder>` asks you to accept the sharing
agreement first, then writes a plain copy to upload. In Colab: `pip install` this repo, run
`better-voice-train --data <folder>`. Back home: `better-voice-train --import-model <folder>`
encrypts the model, then delete every plain copy.

**3. Put it in the app.**

```bash
better-voice-export
```

Then in Better Voice open **Models**, press **Rescan** if the app was open, and pick **My voice**.

**Keep improving it:** set `"keep_audio": true` and each transcription from `--serve` is saved with the
model's guess. `better-voice-record --review` lets you confirm or correct each one, then train
again.

## Use it from your own apps

```bash
better-voice --serve            # http://127.0.0.1:8765/transcribe
```

POST audio (WAV/FLAC/OGG) with `Authorization: Bearer <server_token from config.json>`. The
reply looks like `{"text": "...", "raw": "...", "seconds": 2.4}`. To reach it from another
device, put it behind a tunnel (for example Cloudflare Tunnel). Never open the port to the
internet directly.

Testing: `better-voice --file memo.wav` transcribes a file;
`better-voice --clean "um so i i think new line we should go"` tries the cleanup rules.

## Settings

`config.json` in `%APPDATA%\better-voice` (Windows), `~/Library/Application Support/better-voice`
(Mac) or `~/.config/better-voice` (Linux). Recordings and your model live in `voice-data/` there.
The desktop app has its own settings window.

| Setting | What it does |
| --- | --- |
| `model` | `UsefulSensors/moonshine-base`, or your trained model's folder (set by training) |
| `device` | `auto`, `cpu`, `cuda` (NVIDIA) or `mps` (Apple Silicon) |
| `dictionary` | Your words: fixes their spelling and adds them to the recording sentences |
| `replacements` | Spoken phrase → text to insert, e.g. `{"my email": "me@example.com"}` |
| `cleanup` | Remove "um", stutters; handle "new line", "new paragraph", "scratch that" |
| `llm_cleanup` | Extra polish with a local [Ollama](https://ollama.com) model |
| `encrypt` | Encrypt your recordings and trained model (default on; see PRIVACY.md) |
| `keep_audio` | Save transcriptions as training data you can correct |
| `history` | Save every transcription's text to `history.jsonl` |
| `server_token` | Password for `--serve`, created on first start |

## Development

```bash
python -m unittest discover -s tests
```

| File | Role |
| --- | --- |
| `better_voice/record.py`, `phone.py` | Recording sentences (computer mic or phone) |
| `better_voice/prompts.py`, `data/` | The sentences, and the general-speech check clips |
| `better_voice/dataset.py`, `vault.py` | Recordings on disk, encrypted with the key in the OS credential store |
| `better_voice/consent.py` | The agreement before voice data leaves the computer |
| `better_voice/train.py`, `wer.py` | Fine-tuning and scoring |
| `better_voice/export.py` | Converting a trained model for the app |
| `better_voice/audio.py` | Mic, audio files, the speech model |
| `better_voice/cleanup.py` | Cleanup rules and the optional AI polish |
| `better_voice/app.py`, `serve.py` | Command line and the speech-to-text server |
| `app/` | The desktop app; see [app/README.md](app/README.md) |

Installers: GitHub **Actions → Release → Run workflow** builds them and attaches them to a
draft release.

## Disclaimer

Better Voice is free software provided "as is", without warranty of any kind, express or
implied, including the warranties of merchantability, fitness for a particular purpose and
non-infringement (see [LICENSE](LICENSE)). You use it at your own risk.

Your recordings, transcripts and trained voice model can be used to recognize or imitate your
voice. Better Voice keeps them on your computer (encrypted, unless you turn that off), but you
are responsible for your computer, your encryption key and every copy you make or share,
including recordings sent from your phone and plain copies made for training elsewhere. Neither
Better Voice nor its authors or contributors are liable for any claim, damages, loss of data or
other liability arising from the software or its use, including anything that happens if your
voice data or voice model is leaked, copied or misused.

Speech-to-text makes mistakes. Check dictated text before you send or rely on it, and don't use
Better Voice where an error could cause harm (medical, legal, financial or safety-critical use).
