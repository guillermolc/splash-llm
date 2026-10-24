# Splash Studio

A native macOS control panel for a local Splash server, built with SwiftUI and
a small AppKit menu bar integration. Studio reuses an existing Splash checkout
and prepared models. It does not download models or change their weights.

## Run from source

From this directory:

```sh
./run.sh
```

The launcher always asks SwiftPM to check source dependencies. Unchanged builds
reuse its incremental cache; changing a Swift source file updates the executable.

To create a double-clickable local application:

```sh
./build-app.sh
open ".build/Splash Studio.app"
```

The bundle is signed locally with an ad hoc signature. Build products remain
inside the ignored `.build/` directory. Building Studio needs the Swift toolchain;
starting a real engine also needs the selected checkout's existing
`.venv/bin/python`, `build/splash`, and prepared model/tokenizer directory.

## Main window

The summary shows the selected/active model, API URL and three indicators:
current generation speed, Metal memory, and active/queued requests.

- **Start / Stop:** control the process Studio starts. Startup can be cancelled.
- **Open chat / Copy API URL:** connect a client to the active endpoint.
- **Activity:** active requests and up to five completed requests.
- **Details:** generation samples, latency, draft acceptance, KV allocation,
  actual SSD usage, weight state, Neural Engine status, and diagnostics.
- **Settings** (`⌘,`): edit a draft, validate it, then save. A running owned
  server keeps its applied settings until **Apply and restart** is selected.
  Interrupting known active requests asks for confirmation.

The window follows the system appearance and scrolls at smaller sizes.
Unavailable and stale metrics are distinguished from zero.

## Settings

Basic settings contain the model, compatible maximum context and memory limit.
Advanced settings contain the port, KV format, Neural Engine, INT8 prefill,
SSD capacity/persistence, idle release and checkout directory.

Existing settings remain at:

```text
~/Library/Application Support/splash-menubar/config.json
```

New fields migrate individually; adding an option does not reset existing
settings. Saving is atomic. A malformed file is preserved and reported until
the user explicitly saves a replacement.

## Process ownership

Studio distinguishes an engine it started from a server it connected to.

- An existing server on the configured port is monitored without taking
  ownership. **Disconnect** leaves that server running.
- Closing the main window keeps Studio and its owned engine running.
- **Quit** waits for an owned engine to stop. It leaves an external server alone.
- Stop/restart target the isolated process group created for the owned engine.
  They do not use a global process-name kill.
- An explicit CLI stop separately verifies the local server PID, owner,
  command and listening port before stopping that server and its descendants.

Network calls are asynchronous. One cancelable status loop ignores responses
from old connections. Model discovery reads prepared manifests and caches the
catalog; it does not repeatedly walk weight files. Log reading is incremental
and bounded.

## Metric definitions

- **Current generation speed:** the valid current decode batch, while decoding.
  It is not a retained last-positive value or the lifetime average.
- **Metal memory:** reported current device allocation, with the engine limit
  shown separately. It is not total host RAM or process RSS.
- **KV:** allocated memory and physical resident capacity. Active and cached
  page counts may overlap and are not added together as conversation length.
- **Context:** the server/model limit per request, distinct from global KV
  capacity. Request input/output counts appear in Activity.
- **SSD:** values reported by the server. A difference between prompt size and
  KV capacity is never presented as conversation compaction or disk residency.
- **History:** at most 120 speed samples. Poll intervals vary with visibility
  and activity, so the graph is labeled in samples rather than a fixed duration.

## One-shot commands

```sh
.build/release/splash-studio --models [--root /path/to/splash]
.build/release/splash-studio --status [--port 8001]
.build/release/splash-studio --release [--port 8001]
.build/release/splash-studio --stop [--port 8001]
.build/release/splash-studio --logs
```

Failures return a nonzero exit status. Memory release uses
`POST /v1/memory/release`; success is checked rather than assumed.
The server can release idle weights even when automatic idle release is off.

## Isolated validation profiles

Set `SPLASH_STUDIO_CONFIG` to an absolute configuration-file path to use a
separate development profile. Its logs go to a sibling `logs/engine.log`.
An invalid override is reported and never falls back to writing normal settings.

The regression suite uses temporary settings, metadata-only model directories
and a small local HTTP fixture. It does not load the 27B model, perform inference
or alter the normal user configuration:

```sh
SPLASH_STUDIO_CONFIG=/tmp/splash-studio-validation/config.json swift test
```

All generated documentation is maintained in English.

[Validation results and captured interface](docs/validation-2026-10-24.md)
record the completed checks and their scope.
