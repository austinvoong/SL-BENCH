"""
backend/core/runner.py

Path context (resolved at import time):
  __file__    → sl-bench/backend/core/runner.py
  BACKEND_DIR → sl-bench/backend/   ← added to sys.path so 'core.db' resolves
  ROOT_DIR    → sl-bench/           ← cwd for subprocess so sl_bench/ is importable
"""

import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone

from bson import ObjectId

# ── Path setup (must come before any local imports) ───────────────────────────
# runner.py is at  sl-bench/backend/core/runner.py
# Going up two levels gives sl-bench/backend/, which contains the 'core' package.
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR    = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Now safe to import at module level — no longer lazy inside _worker
from core.db import get_db  # noqa: E402

EXPERIMENT_SCRIPT = os.path.join(ROOT_DIR, "run_experiment.py")

# Hard ceiling: 3 minutes per epoch. A 50-epoch run times out at 150 min.
SECONDS_PER_EPOCH = 180


class RunQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._current: dict | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._thread.start()
        print("[runner] Background worker started", flush=True)

    def enqueue(self, run_id: str, params: dict):
        self._queue.put({"run_id": run_id, "params": params})
        print(f"[runner] Queued run {run_id} ({params['attack']} vs {params['defense']})", flush=True)

    def status(self) -> dict:
        with self._lock:
            return {
                "current": self._current,
                "queued": self._queue.qsize(),
            }

    # ── internal ──────────────────────────────────────────────────────────────

    def _worker(self):
        """Infinite blocking loop — one job at a time."""
        while True:
            job = self._queue.get()
            run_id = job["run_id"]
            params = job["params"]

            with self._lock:
                self._current = {"run_id": run_id, "params": params}

            db = get_db()
            db.runs.update_one(
                {"_id": ObjectId(run_id)},
                {"$set": {"status": "running", "started_at": datetime.now(timezone.utc)}},
            )
            print(f"[runner] Starting run {run_id}", flush=True)

            try:
                result = self._execute(run_id, params)
                db.runs.update_one(
                    {"_id": ObjectId(run_id)},
                    {"$set": {
                        "status": "complete",
                        "finished_at": datetime.now(timezone.utc),
                        **result,
                    }},
                )
                print(f"[runner] Completed run {run_id}: SSIM={result.get('ssim')}", flush=True)

            except subprocess.TimeoutExpired:
                msg = f"Timed out after {params.get('epochs', 10) * SECONDS_PER_EPOCH}s"
                db.runs.update_one(
                    {"_id": ObjectId(run_id)},
                    {"$set": {"status": "failed", "finished_at": datetime.now(timezone.utc), "error": msg}},
                )
                print(f"[runner] TIMEOUT run {run_id}: {msg}", flush=True)

            except Exception as exc:
                db.runs.update_one(
                    {"_id": ObjectId(run_id)},
                    {"$set": {"status": "failed", "finished_at": datetime.now(timezone.utc), "error": str(exc)}},
                )
                print(f"[runner] FAILED run {run_id}: {exc}", flush=True)

            finally:
                with self._lock:
                    self._current = None
                self._queue.task_done()

    def _execute(self, run_id: str, params: dict) -> dict:
        """
        Spawn run_experiment.py as a child process.

        cwd=ROOT_DIR means run_experiment.py's own sys.path.insert(0, ROOT_DIR)
        correctly picks up sl_bench/. All progress prints are ignored;
        only the final JSON line is parsed for results.
        stderr is always captured and stored for debugging.
        """
        epochs = params.get("epochs", 10)
        timeout = epochs * SECONDS_PER_EPOCH

        cmd = [
            sys.executable, EXPERIMENT_SCRIPT,
            "--attack",       params["attack"],
            "--defense",      params["defense"],
            "--architecture", params["architecture"],
            "--cut_layer",    str(params["cut_layer"]),
            "--epochs",       str(epochs),
            "--run_id",       run_id,
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT_DIR,
        )

        # Always log stderr for debugging
        if proc.stderr:
            print(f"[runner][{run_id}] stderr:\n{proc.stderr[-3000:]}", flush=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"run_experiment.py exited {proc.returncode}.\n"
                f"stderr (last 2000 chars):\n{proc.stderr[-2000:]}"
            )

        # Parse the last valid JSON line that contains our expected keys
        required = {"ssim", "psnr", "dcor", "accuracy"}
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if required.issubset(data.keys()):
                    return {k: float(data[k]) for k in required} | {
                        "note": data.get("note", ""),
                        "raw_output": proc.stdout[-4000:],
                    }
            except (json.JSONDecodeError, ValueError):
                continue

        raise RuntimeError(
            f"No valid result JSON found in script output.\n"
            f"stdout (last 2000 chars):\n{proc.stdout[-2000:]}"
        )