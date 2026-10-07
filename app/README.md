# Better Voice desktop app

The Better Voice app: press a shortcut, talk, and the text is typed into whatever app you're in.
It runs a speech model on your own computer, including one trained on **your** voice with the
Python tools in the repo root (`better-voice-record`, `better-voice-train`, `better-voice-export`).

Built on [Handy](https://github.com/cjpais/Handy) by CJ Pais (MIT license, see `LICENSE`).
Handy's own README is kept as `README.handy.md`, and its development notes in `AGENTS.md` and
`BUILD.md` apply here unchanged.

## Use your own voice model

1. Train it (repo root): `better-voice-record`, then `better-voice-train`.
2. Convert it for the app: `pip install -e ".[export]"`, then `better-voice-export`.
   This writes `my-voice-moonshine/` into the app's models folder:
   - Windows: `%APPDATA%\com.bettervoice.app\models`
   - macOS: `~/Library/Application Support/com.bettervoice.app/models`
   - Linux: `~/.local/share/com.bettervoice.app/models`
3. In Better Voice, open **Models**, press **Rescan** if the app was already open, and pick
   **My voice**.

The app runs Moonshine Base, so train from `UsefulSensors/moonshine-base` (the default).

## Develop

```bash
bun install
mkdir -p src-tauri/resources/models   # the voice detection model is already committed
bun run tauri dev
```

Checks before committing: `bun run lint`, `bun run check:translations`, `cargo fmt` and
`cargo test` in `src-tauri/`.

## What changed from Handy

- Name, app id (`com.bettervoice.app`), icons, tray icons, logo and About page.
- Updates come from Better Voice's own releases, signed with its own key (the repository
  secret `TAURI_SIGNING_PRIVATE_KEY`; `UPDATE_FEED_READY` in `src-tauri/src/settings.rs` turns
  the updater off again). Handy's Windows code signing is removed.
- Custom Moonshine models in the models folder show up as "My voice"
  (`discover_custom_moonshine_models` in `src-tauri/src/managers/model.rs`).
- Stock models still download from Handy's model host (blob.handy.computer).
