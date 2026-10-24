"""Local HTTP lifecycle fixture. It never imports or runs the inference engine."""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = pathlib.Path.cwd()
CONTROL = ROOT / "fixture.json"
LOCK = threading.Lock()
REQUESTS = 0
RELEASED = False


def control():
    try:
        return json.loads(CONTROL.read_text())
    except (OSError, ValueError):
        return {}


def record(name, value):
    target = ROOT / name
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    args, _ = parser.parse_known_args()
    initial = control()
    time.sleep(initial.get("startup_delay", 0))
    if initial.get("fail_start"):
        print("Error · fixture startup failed", flush=True)
        raise SystemExit(7)

    child = None
    if initial.get("spawn_child"):
        child = subprocess.Popen(
            [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)",
             "splash-studio-fixture-child", str(ROOT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        record("child.json", {"pid": child.pid, "group": os.getpgrp()})

    instance = f"fixture-{os.getpid()}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, code, body):
            payload = json.dumps(body).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            global REQUESTS
            if self.path != "/status":
                self.send_json(404, {"error": {"message": "unknown fixture route"}})
                return
            with LOCK:
                REQUESTS += 1
                count = REQUESTS
                record("status-requests.json", {"count": count})
            time.sleep(control().get("status_delay", 0))
            settings = control()
            self.send_json(200, {
                "schema_version": 6,
                "ready": settings.get("ready", True),
                "instance": {"id": instance, "pid": os.getpid(), "model": args.model, "port": args.port},
                "requests": {"submitted": count, "completed": count, "cancelled": 0, "failed": 0},
                "scheduler": {"decoding": 0, "prefilling": 0, "queued": 0},
                "active_requests": [],
                "maximum_context_tokens": 32768,
                "memory_pressure": settings.get("memory_pressure", "normal"),
                "metal": {"healthy": settings.get("metal_healthy", True),
                          "failure_reason": settings.get("metal_failure_reason", "")},
                "weights": {"released": RELEASED, "restoring": False, "idle_release_seconds": 0},
            })

        def do_POST(self):
            global RELEASED
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            record("release-requests.json", {"path": self.path})
            code = control().get("release_status", 200)
            if self.path != "/v1/memory/release":
                self.send_json(404, {"error": {"message": "wrong memory release route"}})
            elif code != 200:
                self.send_json(code, {"error": {"message": "fixture release refused"}})
            else:
                RELEASED = True
                self.send_json(200, {"released": True, "weights": {"released": True, "restoring": False}})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    def watch_exit():
        while True:
            time.sleep(0.05)
            if control().get("exit_now"):
                os._exit(7)

    threading.Thread(target=watch_exit, daemon=True).start()

    def interrupt(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    print(f"Ready · fixture · {args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if child is not None:
            child.kill()
            child.wait(timeout=2)


if __name__ == "__main__":
    main()
