"""Early, append-only startup diagnostics, before model or distributed setup."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def startup_event(config, event, **fields):
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    worker = str(rank) if "RANK" in os.environ else "launcher"
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
              "rank": rank, "worker": worker, "pid": os.getpid(), **fields}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    directory = output / "startup"
    directory.mkdir(exist_ok=True)
    with (directory / f"{worker}.jsonl").open("a", encoding="utf8") as stream:
        stream.write(line)
    if rank == 0:
        with (output / "events.jsonl").open("a", encoding="utf8") as stream:
            stream.write(line)
        print(record["time"], event, " ".join(f"{key}={value}" for key, value in fields.items()), flush=True)


@contextmanager
def startup_stage(config, stage, **fields):
    """Heartbeats report elapsed wait, never claim the operation is progressing."""
    started = time.monotonic()
    interval = max(0.1, float(config.get("logging", {}).get("progress_interval_seconds", 10)))
    stopped = threading.Event()
    startup_event(config, "startup_started", stage=stage, **fields)

    def heartbeat():
        while not stopped.wait(interval):
            startup_event(config, "startup_waiting", stage=stage,
                          elapsed_seconds=round(time.monotonic() - started, 2))

    thread = threading.Thread(target=heartbeat, name=f"startup-{stage}", daemon=True)
    thread.start()
    try:
        yield
    except BaseException as error:
        stopped.set()
        thread.join()
        startup_event(config, "startup_failed", stage=stage,
                      elapsed_seconds=round(time.monotonic() - started, 2),
                      error=f"{type(error).__name__}: {error}")
        raise
    else:
        stopped.set()
        thread.join()
        startup_event(config, "startup_finished", stage=stage,
                      elapsed_seconds=round(time.monotonic() - started, 2))


@contextmanager
def capture_console(config):
    """Capture Python, native-library and inherited child output with Unix tee."""
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "console.log"
    if os.environ.get("DRAW_ALIGN_CONSOLE_LOG_ACTIVE") == "1":
        yield path
        return
    # Fail before launching any model work if the destination is not writable.
    with path.open("ab"):
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    previous = os.environ.get("DRAW_ALIGN_CONSOLE_LOG_ACTIVE")
    tee = None
    try:
        tee = subprocess.Popen(["tee", "-a", str(path)], stdin=subprocess.PIPE,
                               stdout=saved_out, stderr=saved_err, start_new_session=True)
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
        os.environ["DRAW_ALIGN_CONSOLE_LOG_ACTIVE"] = "1"
        yield path
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        if previous is None:
            os.environ.pop("DRAW_ALIGN_CONSOLE_LOG_ACTIVE", None)
        else:
            os.environ["DRAW_ALIGN_CONSOLE_LOG_ACTIVE"] = previous
        if tee is not None:
            tee.stdin.close()
            try:
                tee.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tee.terminate()
                tee.wait(timeout=5)
