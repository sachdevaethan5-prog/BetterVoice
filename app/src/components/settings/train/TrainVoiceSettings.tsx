// Better Voice: record your voice, train your own model and switch the app to it.
// The work is done by the Python voice tools in the user's better-voice folder
// (see src-tauri/src/commands/voice_tools.rs); this page starts them and shows their output.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import { openUrl } from "@tauri-apps/plugin-opener";
import { toast } from "sonner";
import { commands } from "@/bindings";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { SettingContainer } from "../../ui/SettingContainer";
import { Button } from "../../ui/Button";
import { Input } from "../../ui/Input";

type Task = "record" | "train" | "export";

interface Status {
  minutes: number;
  recordings: number;
  sentences_left: number;
  min_minutes: number;
  trained: boolean;
  using_trained_model: boolean;
  encrypted: boolean;
}

const GOAL_MINUTES = 30;
const MY_VOICE_MODEL = "my-voice-moonshine"; // the folder better-voice-export writes
const FOLDER_KEY = "betterVoice.toolsFolder";
const RECORDER_URL = /http:\/\/127\.0\.0\.1:\d+\/[\w-]+\//;

const savedFolder = () => {
  try {
    return localStorage.getItem(FOLDER_KEY) ?? "";
  } catch {
    return "";
  }
};

export const TrainVoiceSettings: React.FC = () => {
  const { t } = useTranslation();
  const [folder, setFolder] = useState(savedFolder);
  const [toolsDir, setToolsDir] = useState<string | null>(null);
  const [findError, setFindError] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const [running, setRunning] = useState<Task | null>(null);
  const [recorderUrl, setRecorderUrl] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const logRef = useRef<HTMLPreElement>(null);

  const refresh = useCallback(async (dir: string) => {
    const r = await commands.voiceToolsStatus(dir);
    if (r.status === "ok") setStatus(JSON.parse(r.data) as Status);
    else toast.error(r.error);
  }, []);

  const find = useCallback(
    async (path: string) => {
      const r = await commands.voiceToolsFind(path || null);
      if (r.status === "ok") {
        setToolsDir(r.data);
        setFindError("");
        refresh(r.data);
      } else {
        setToolsDir(null);
        setFindError(r.error);
      }
    },
    [refresh],
  );

  useEffect(() => {
    find(savedFolder());
  }, [find]);

  useEffect(() => {
    const offLine = listen<{ task: Task; line: string }>(
      "voice-tools-line",
      ({ payload }) => {
        const url = payload.line.match(RECORDER_URL);
        if (payload.task === "record" && url) setRecorderUrl(url[0]);
        else setLog((l) => [...l.slice(-300), payload.line]);
      },
    );
    const offDone = listen<{ task: Task; code: number | null }>(
      "voice-tools-done",
      async ({ payload }) => {
        setRunning(null);
        setRecorderUrl("");
        if (payload.task === "export" && payload.code === 0) {
          await commands.rescanLocalModels();
          const r = await commands.setActiveModel(MY_VOICE_MODEL);
          if (r.status === "ok")
            toast.success(t("settings.trainVoice.use.done"));
          else toast.error(r.error);
        }
        if (toolsDir) refresh(toolsDir);
      },
    );
    return () => {
      offLine.then((f) => f());
      offDone.then((f) => f());
    };
  }, [refresh, t, toolsDir]);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [log]);

  // A run carries on while you use the rest of the app; pick it back up when the page opens.
  // The recorder is the exception: it needs this page, so leaving the page stops it.
  useEffect(() => {
    commands.voiceToolsRunning().then((task) => {
      if (task === "record") commands.voiceToolsStop();
      else if (task) setRunning(task as Task);
    });
    return () => {
      commands.voiceToolsRunning().then((task) => {
        if (task === "record") commands.voiceToolsStop();
      });
    };
  }, []);

  const start = async (task: Task) => {
    if (!toolsDir) return;
    setLog([]);
    setRecorderUrl("");
    const r = await commands.voiceToolsStart(toolsDir, task);
    if (r.status === "ok") setRunning(task);
    else toast.error(r.error);
  };
  const stop = async () => {
    await commands.voiceToolsStop();
    setRunning(null);
  };

  const saveFolder = () => {
    try {
      localStorage.setItem(FOLDER_KEY, folder);
    } catch {
      // the folder just isn't remembered
    }
    find(folder);
  };

  const enough = status ? status.minutes >= status.min_minutes : false;
  const pct = (m: number) => `${Math.min(100, (m / GOAL_MINUTES) * 100)}%`;

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <p className="text-sm text-mid-gray px-4">
        {t("settings.trainVoice.intro")}
      </p>

      <SettingsGroup title={t("settings.trainVoice.tools.title")}>
        <SettingContainer
          title={t("settings.trainVoice.tools.folder")}
          description={t("settings.trainVoice.tools.folderDescription")}
          grouped={true}
          layout="stacked"
        >
          <div className="flex gap-2 flex-wrap">
            <Input
              id="voice-tools-folder"
              className="flex-1 min-w-0"
              value={folder}
              placeholder={t("settings.trainVoice.tools.placeholder")}
              onChange={(e) => setFolder(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveFolder()}
            />
            <Button variant="secondary" onClick={saveFolder}>
              {t("settings.trainVoice.tools.find")}
            </Button>
          </div>
          <p
            className={`text-xs mt-2 ${toolsDir ? "text-mid-gray" : "text-warning"}`}
          >
            {toolsDir
              ? t("settings.trainVoice.tools.found", { dir: toolsDir })
              : findError && t("settings.trainVoice.tools.notFound")}
          </p>
        </SettingContainer>
      </SettingsGroup>

      {toolsDir && status && (
        <>
          <SettingsGroup title={t("settings.trainVoice.progress.title")}>
            <div className="px-4 py-3 space-y-2">
              <div className="flex items-baseline justify-between gap-4 flex-wrap">
                <span className="text-sm font-semibold tabular-nums">
                  {t("settings.trainVoice.progress.minutes", {
                    minutes: status.minutes.toFixed(1),
                    goal: GOAL_MINUTES,
                  })}
                </span>
                <span className="text-xs text-mid-gray tabular-nums">
                  {t("settings.trainVoice.progress.recordings", {
                    count: status.recordings,
                  })}
                </span>
              </div>
              <div className="relative h-2 rounded-full bg-mid-gray/20">
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-logo-primary"
                  style={{ width: pct(status.minutes) }}
                />
                <div
                  className="absolute -top-1 -bottom-1 w-0.5 bg-text/60"
                  style={{ left: pct(status.min_minutes) }}
                  title={t("settings.trainVoice.progress.minimum", {
                    minutes: status.min_minutes,
                  })}
                />
              </div>
              <p className="text-xs text-mid-gray">
                {enough
                  ? t("settings.trainVoice.progress.ready")
                  : t("settings.trainVoice.progress.needMore", {
                      minutes: (status.min_minutes - status.minutes).toFixed(1),
                    })}{" "}
                {status.encrypted
                  ? t("settings.trainVoice.progress.private")
                  : t("settings.trainVoice.progress.notEncrypted")}
              </p>
            </div>
          </SettingsGroup>

          <SettingsGroup title={t("settings.trainVoice.steps.title")}>
            <SettingContainer
              title={t("settings.trainVoice.record.title")}
              description={t("settings.trainVoice.record.description")}
              grouped={true}
            >
              {running === "record" ? (
                <Button variant="secondary" onClick={stop}>
                  {t("settings.trainVoice.record.stop")}
                </Button>
              ) : (
                <Button onClick={() => start("record")} disabled={!!running}>
                  {t("settings.trainVoice.record.start")}
                </Button>
              )}
            </SettingContainer>
            {running === "record" && recorderUrl && (
              <div className="px-4 py-3 space-y-2">
                <iframe
                  title={t("settings.trainVoice.record.title")}
                  src={recorderUrl}
                  allow="microphone"
                  className="w-full h-[480px] rounded-lg border border-mid-gray/20"
                />
                <p className="text-xs text-mid-gray">
                  {t("settings.trainVoice.record.browserHint")}{" "}
                  <button
                    type="button"
                    className="underline hover:text-logo-primary"
                    onClick={() => openUrl(recorderUrl)}
                  >
                    {t("settings.trainVoice.record.openBrowser")}
                  </button>
                </p>
              </div>
            )}

            <SettingContainer
              title={t("settings.trainVoice.train.title")}
              description={
                enough
                  ? t("settings.trainVoice.train.description")
                  : t("settings.trainVoice.train.locked", {
                      minutes: status.min_minutes,
                    })
              }
              grouped={true}
            >
              {running === "train" ? (
                <Button variant="secondary" onClick={stop}>
                  {t("settings.trainVoice.train.stop")}
                </Button>
              ) : (
                <Button
                  onClick={() => start("train")}
                  disabled={!enough || !!running}
                >
                  {t("settings.trainVoice.train.start")}
                </Button>
              )}
            </SettingContainer>

            <SettingContainer
              title={t("settings.trainVoice.use.title")}
              description={
                status.trained
                  ? t("settings.trainVoice.use.description")
                  : t("settings.trainVoice.use.locked")
              }
              grouped={true}
            >
              <Button
                onClick={() => start("export")}
                disabled={!status.trained || !!running}
              >
                {running === "export"
                  ? t("settings.trainVoice.use.working")
                  : t("settings.trainVoice.use.start")}
              </Button>
            </SettingContainer>
          </SettingsGroup>

          {log.length > 0 && (
            <SettingsGroup title={t("settings.trainVoice.log")}>
              <pre
                ref={logRef}
                className="px-4 py-3 text-xs font-mono whitespace-pre-wrap max-h-64 overflow-y-auto"
              >
                {log.join("\n")}
              </pre>
            </SettingsGroup>
          )}
        </>
      )}
    </div>
  );
};
