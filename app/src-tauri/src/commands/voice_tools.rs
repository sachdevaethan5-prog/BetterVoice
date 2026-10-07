//! Better Voice: the "Train my voice" page runs the Python voice tools (better-voice-record,
//! better-voice-train, better-voice-export) from the user's better-voice folder and streams
//! their output to the page. Only these three tools, with fixed arguments, can be started.

use serde::Serialize;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter};

/// The one tool run at a time (recording, training or exporting), tagged with its run number
/// so a finished run never touches the one started after it.
static RUNNING: Mutex<Option<(u64, String, Child)>> = Mutex::new(None);
static RUNS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Serialize)]
struct Line {
    task: String,
    line: String,
}

#[derive(Clone, Serialize)]
struct Done {
    task: String,
    code: Option<i32>,
}

fn exe(dir: &Path, name: &str) -> PathBuf {
    if cfg!(windows) {
        dir.join(format!("{name}.exe"))
    } else {
        dir.join(name)
    }
}

/// Where the tools are inside a better-voice folder (its virtual environment), or the folder itself.
fn scripts_dir(folder: &Path) -> Option<PathBuf> {
    [
        folder.join(".venv").join("Scripts"),
        folder.join(".venv").join("bin"),
        folder.join("Scripts"), // the virtual environment folder itself
        folder.join("bin"),
        folder.to_path_buf(),
    ]
    .into_iter()
    .find(|d| exe(d, "better-voice-record").is_file())
}

fn command(dir: &Path, name: &str, args: &[&str]) -> Command {
    let mut cmd = Command::new(exe(dir, name));
    cmd.args(args)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flashing up
    }
    cmd
}

/// The tools folder: the one given, else `better-voice` in the home folder. `Err` explains
/// what's missing.
#[tauri::command]
#[specta::specta]
pub fn voice_tools_find(folder: Option<String>) -> Result<String, String> {
    let candidates: Vec<PathBuf> = match folder.filter(|f| !f.trim().is_empty()) {
        Some(f) => vec![PathBuf::from(f.trim())],
        None => std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(|h| vec![PathBuf::from(h).join("better-voice")])
            .unwrap_or_default(),
    };
    candidates
        .iter()
        .find_map(|f| scripts_dir(f))
        .map(|d| d.to_string_lossy().into_owned())
        .ok_or_else(|| "The Better Voice voice tools weren't found there.".to_string())
}

/// Recordings and model status, as the JSON `better-voice-record --status` prints.
#[tauri::command]
#[specta::specta]
pub async fn voice_tools_status(dir: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let out = command(Path::new(&dir), "better-voice-record", &["--status"])
            .output()
            .map_err(|e| format!("Couldn't run the voice tools: {e}"))?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
        } else {
            Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Start `task` ("record", "train" or "export"), stopping any tool already running. Output
/// arrives as `voice-tools-line` events, the end as one `voice-tools-done` event.
#[tauri::command]
#[specta::specta]
pub fn voice_tools_start(app: AppHandle, dir: String, task: String) -> Result<(), String> {
    let (name, args): (&str, &[&str]) = match task.as_str() {
        "record" => ("better-voice-record", &["--local"]),
        "train" => ("better-voice-train", &[]),
        "export" => ("better-voice-export", &[]),
        _ => return Err(format!("Unknown task {task}")),
    };
    voice_tools_stop();
    let mut child = command(Path::new(&dir), name, args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Couldn't start {name}: {e}"))?;
    let readers: Vec<Box<dyn Read + Send>> = vec![
        Box::new(child.stdout.take().unwrap()),
        Box::new(child.stderr.take().unwrap()),
    ];
    let run = RUNS.fetch_add(1, Ordering::SeqCst) + 1;
    *RUNNING.lock().unwrap() = Some((run, task.clone(), child));
    let mut threads = Vec::new();
    for reader in readers {
        let (app, task) = (app.clone(), task.clone());
        threads.push(std::thread::spawn(move || {
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                let _ = app.emit(
                    "voice-tools-line",
                    Line {
                        task: task.clone(),
                        line,
                    },
                );
            }
        }));
    }
    std::thread::spawn(move || {
        for t in threads {
            let _ = t.join(); // both pipes closed: the tool has exited
        }
        let mine = {
            let mut running = RUNNING.lock().unwrap();
            match running.as_ref() {
                Some((id, _, _)) if *id == run => running.take(),
                _ => None, // stopped, or replaced by a newer run
            }
        };
        let code = mine
            .and_then(|(_, _, mut c)| c.wait().ok())
            .and_then(|s| s.code());
        let _ = app.emit("voice-tools-done", Done { task, code });
    });
    Ok(())
}

/// Stop the running tool, if any (for example, closing the recorder).
#[tauri::command]
#[specta::specta]
pub fn voice_tools_stop() {
    if let Some((_, _, mut child)) = RUNNING.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Which tool is running ("record", "train", "export"), if any; the page asks when it opens,
/// since a long training run carries on while you use the rest of the app.
#[tauri::command]
#[specta::specta]
pub fn voice_tools_running() -> Option<String> {
    RUNNING
        .lock()
        .unwrap()
        .as_ref()
        .map(|(_, task, _)| task.clone())
}
