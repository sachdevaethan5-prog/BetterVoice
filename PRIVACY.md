# Your voice data

Your recordings, their transcripts and the model trained on them can be used to recognize or
imitate your voice. Better Voice treats them as private.

## Where it lives

Only on your computer, in `voice-data/` inside the Better Voice settings folder
(`%APPDATA%\better-voice` on Windows, `~/Library/Application Support/better-voice` on Mac,
`~/.config/better-voice` on Linux). Nothing is uploaded, synced or sent anywhere unless you use
one of the two features below and accept the agreement first.

The desktop app's copy of your model (made by `better-voice-export`) lives in the app's own
folder and is a normal, unencrypted file, so the app can load it like any other model.

## Encryption

On by default (`"encrypt": true` in `config.json`):

- Each recording, the list of transcripts, and your trained model's weights are encrypted with
  a random key (Fernet: AES-128 with an HMAC check).
- The key is kept in your system's credential store (Windows Credential Manager, macOS Keychain,
  Linux Secret Service), never in a file. Data is only decrypted in memory.
- It protects copies of the files: backups, cloud sync, other people using the computer, a lost
  or stolen disk. It can't protect against someone, or malware, already signed in as you, since
  they can read the key the same way Better Voice does. Whole-disk encryption (BitLocker,
  FileVault) adds another layer.
- **Save the key.** If it's lost (new computer, wiped credential store), your recordings can't be
  read. `better-voice-record --show-key` prints it: keep it only in a password manager. To use
  it on another computer, set `BETTER_VOICE_KEY` to it.
- Turning encryption off (`"encrypt": false`) decrypts everything back to plain files the next
  time you run a command; turning it on again re-encrypts them.

## Features that send voice data off your computer

Each asks you to accept this agreement first (once per version of the agreement):

> Your recordings, transcripts and trained voice model can be used to recognize or imitate your
> voice. Once they leave this computer, Better Voice can't protect that copy or that connection.
> You are responsible for where it goes and who can get it. Better Voice and its authors are not
> responsible for anything that happens if your voice data or voice model is leaked, copied or
> misused once it leaves this computer, and give no warranty of any kind (see the MIT license).

- **`better-voice-record --phone`**: your audio goes from the phone through Cloudflare's tunnel
  (trycloudflare.com) to your computer.
- **`better-voice-record --pack DIR`**: writes a plain, unencrypted copy of your recordings for
  training on another computer such as Google Colab. Bring the result home with
  `better-voice-train --import-model <folder>` (encrypted on arrival), then delete every plain
  copy, including any you uploaded.

`better-voice --serve` (for your own apps) receives audio and returns text; your model files never
leave your computer.
