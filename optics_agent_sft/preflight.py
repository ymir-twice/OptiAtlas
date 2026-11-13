"""Bounded, observable subprocess supervision for the local CUDA probe.

This module uses only the standard library so launch plans remain usable without
the training environment. Child output is retained in a log; parent heartbeats
also cover interpreter startup, before the child can print any progress.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


EVENT_PREFIX = "OPTICS_SFT_PREFLIGHT "


class PreflightInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__(f"Local CUDA preflight interrupted by {signal.Signals(signum).name}")
        self.signum = signum


def validate_timing(timeout_seconds, progress_interval):
    for name, value in (("timeout_seconds", timeout_seconds), ("progress_interval", progress_interval)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Preflight {name} must be positive and finite")


def _signal_owned(process, signum):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        else:
            process.send_signal(signum)
    except ProcessLookupError:
        pass


def _reap(process):
    if process is None:
        return
    _signal_owned(process, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A child import can start descendants. Clean the owned group even if
        # its interpreter already exited, without touching other GPU jobs.
        if os.name == "posix":
            _signal_owned(process, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # An uninterruptible kernel wait cannot be made synchronous here.
            print(f"Local CUDA preflight: probe pid={process.pid} did not exit after termination", file=sys.stderr, flush=True)


def _diagnostic_tail(log):
    with log.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 8000))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-20:])


def run_probe(argv, *, environment, log_path, timeout_seconds=1200, progress_interval=10):
    """Return an explicit success report only after a clean probe process exit."""
    validate_timing(timeout_seconds, progress_interval)
    log = Path(log_path).expanduser().absolute()
    process = None
    interrupted = None
    previous = {}
    stage = "interpreter startup"
    started = last_progress = time.monotonic()
    ready = False
    report = None
    pending = ""

    def progress(message):
        print(f"Local CUDA preflight: {message}; log={log}", file=sys.stderr, flush=True)

    def on_signal(signum, _frame):
        nonlocal interrupted
        # Do not raise inside Popen: its child may exist before the handle has
        # been returned. Defer cancellation until cleanup can reach that child.
        if interrupted is None:
            interrupted = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, *([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])):
            previous[signum] = signal.signal(signum, on_signal)
        if interrupted is not None:
            raise PreflightInterrupted(interrupted)
        log.parent.mkdir(parents=True, exist_ok=True)
        progress(f"starting python={argv[0]} timeout={timeout_seconds:g}s")
        with log.open("w", encoding="utf-8") as writer:
            writer.write(f"python={argv[0]}\ntimeout_seconds={timeout_seconds}\n")
            writer.flush()
            if interrupted is not None:
                raise PreflightInterrupted(interrupted)
            try:
                process = subprocess.Popen(list(argv), env=dict(environment), shell=False,
                    stdout=writer, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            except OSError as error:
                writer.write(f"Interpreter launch failed: {type(error).__name__}: {error}\n")
                writer.flush()
                raise RuntimeError(f"Local CUDA preflight could not start interpreter {argv[0]}: "
                                   f"{type(error).__name__} (errno={error.errno}); log={log}") from None
            with log.open("r", encoding="utf-8", errors="replace") as reader:
                def drain():
                    nonlocal pending, stage, ready, report, last_progress
                    pending += reader.read()
                    lines = pending.split("\n")
                    pending = lines.pop()
                    for line in lines:
                        if not line.startswith(EVENT_PREFIX):
                            continue
                        try:
                            event = json.loads(line[len(EVENT_PREFIX):])
                        except ValueError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("event") == "runtime_started":
                            ready = True
                        elif event.get("event") == "stage" and isinstance(event.get("stage"), str):
                            stage = event["stage"]
                            progress(f"checking {stage}")
                            last_progress = time.monotonic()
                        elif event.get("event") == "result":
                            report = event.get("report")

                while True:
                    drain()
                    if interrupted is not None:
                        progress(f"interrupted while checking {stage}; stopping probe pid={process.pid}")
                        raise PreflightInterrupted(interrupted)
                    code = process.poll()
                    if code is not None:
                        drain()
                        break
                    now = time.monotonic()
                    elapsed = now - started
                    if elapsed >= timeout_seconds:
                        if ready and os.name == "posix" and hasattr(signal, "SIGUSR1"):
                            try:
                                os.kill(process.pid, signal.SIGUSR1)
                                time.sleep(.2)
                            except ProcessLookupError:
                                pass
                        raise RuntimeError(f"Local CUDA preflight timed out after {timeout_seconds:g}s "
                                           f"while checking {stage}; log={log}. "
                                           "No training started. Increase --preflight-timeout-seconds "
                                           "if environment imports or CUDA initialization need more time.")
                    if now - last_progress >= progress_interval:
                        progress(f"waiting for {stage}; elapsed={elapsed:.1f}s / {timeout_seconds:g}s")
                        last_progress = now
                    time.sleep(min(.1, progress_interval, max(.001, timeout_seconds - elapsed)))
                if not isinstance(report, dict) or report.get("ok") is not True or code != 0:
                    problem = (report.get("error") if isinstance(report, dict) else None) or "no valid success report"
                    raise RuntimeError(f"Local CUDA preflight failed while checking {stage}: {problem}; "
                                       f"exit_code={code}; log={log}\n{_diagnostic_tail(log)}")
        elapsed = round(time.monotonic() - started, 3)
        progress(f"passed in {elapsed:g}s")
        return {**report, "log": str(log), "elapsed_seconds": elapsed}
    finally:
        try:
            _reap(process)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if interrupted is not None:
            # A signal can arrive after the last poll, during probe cleanup.
            # Do not allow that completed check to proceed into training.
            raise PreflightInterrupted(interrupted)
