# Splash menubar (macOS)

A small menu bar app to run and watch a splash engine from your fork:

- starts/stops the server (spawns `.venv/bin/python -m server.server`)
- picks any model under `install/models`
- context / KV format / idle-release / memory cap / ANE / int8 prefill / SSD cache
- live status (memory, KV, decode speed) from the engine's `/status`
- one-click "release memory" (frees weights, keeps KV)

Build and run:

```sh
./run.sh            # or: swift build -c release && .build/release/splash-menubar
```

Configuration lives in `~/Library/Application Support/splash-menubar/config.json`
(point `root` at your splash-llm checkout). Engine logs land in the `logs/` folder
there. No orphans: quitting the app stops the engine it started.

CLI helpers (same binary):

```sh
.build/release/splash-menubar --models        # installed models + sizes
.build/release/splash-menubar --status [port] # human-readable /status
.build/release/splash-menubar --release [port] # ask the engine to free weights
```
