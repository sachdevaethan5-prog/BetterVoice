"""Encryption for your voice data (recordings, transcripts and your trained model's weights).

One random key per computer, kept in the operating system's credential store (Windows
Credential Manager, macOS Keychain, Linux Secret Service), never in a file. Data is only
ever decrypted in memory. This protects copies of the files (backups, sync, other users of
the computer, a lost disk); it can't protect against someone or something already logged in
as you, which can read the key the same way this program does.

If the key is lost (new computer, wiped credential store), the encrypted data can't be read.
Save it in a password manager: better-voice-record --show-key
"""

from __future__ import annotations

import os

SERVICE, NAME = "better-voice", "voice-data-key"
ENV = "BETTER_VOICE_KEY"  # for computers without a credential store (Colab, headless Linux)


class VaultError(RuntimeError):
    pass


_key: bytes | None = None


def key() -> bytes:
    global _key
    if _key:
        return _key
    if os.environ.get(ENV):
        _key = os.environ[ENV].encode()
        return _key
    import keyring
    from cryptography.fernet import Fernet

    try:
        stored = keyring.get_password(SERVICE, NAME)
        if not stored:
            stored = Fernet.generate_key().decode()
            keyring.set_password(SERVICE, NAME, stored)
            if keyring.get_password(SERVICE, NAME) != stored:  # never encrypt with a key we can't get back
                raise VaultError("The credential store didn't keep the new key.")
    except VaultError:
        raise
    except Exception as exc:
        raise VaultError(
            f"No secure place to keep the encryption key on this computer ({exc}). On Linux, install "
            f"and unlock a keyring (e.g. gnome-keyring), or set {ENV} to a key from a password manager."
        ) from None
    _key = stored.encode()
    return _key


def encrypt(data: bytes) -> bytes:
    from cryptography.fernet import Fernet

    return Fernet(key()).encrypt(data)


def decrypt(data: bytes) -> bytes:
    from cryptography.fernet import Fernet, InvalidToken

    try:
        return Fernet(key()).decrypt(data)
    except InvalidToken:
        raise VaultError("This voice data was encrypted with a different key. Restore the key you saved "
                         f"(set {ENV}) or record again.") from None
