"""The agreement you accept before any of your voice data leaves this computer.

Everything else in Better Voice keeps your recordings and your model on this computer
(encrypted unless the `encrypt` setting is off). Two features can't: recording with your
phone (the audio travels through Cloudflare's tunnel) and packing your recordings for
training elsewhere (a plain copy).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .config import Config

VERSION = 1
TEXT = """\
Your voice data is about to leave this computer.

{what}

Your recordings, transcripts and trained voice model can be used to recognize or imitate
your voice. Better Voice keeps them on this computer (encrypted, unless you turned that
off), but from this point it can't protect this copy or this connection. You are
responsible for where it goes and who can get it. Better Voice and its authors are not
responsible for anything that happens if your voice data or voice model is leaked, copied
or misused once it leaves this computer, and give no warranty of any kind (see the MIT
license).

Type I AGREE to continue, or press Enter to cancel: """


def require(cfg: Config, what: str, ask=input) -> bool:
    """True when the person has accepted the agreement (now or before). Asks every new version."""
    if cfg.sharing_agreement.get("version") == VERSION:
        print("Reminder: you accepted the voice data sharing agreement on "
              f"{cfg.sharing_agreement.get('accepted_at', '?')[:10]}.\n")
        return True
    if ask(TEXT.format(what=what)).strip().upper() != "I AGREE":
        print("Cancelled. Nothing left this computer.")
        return False
    cfg.sharing_agreement = {"version": VERSION, "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    cfg.save()
    return True
