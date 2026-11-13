"""Fixed-size torchrun launch plans and process supervision; no Torch import here."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LaunchPlan:
    argv: tuple[str, ...]
    config_path: str
    config_sha256: str
    python_executable: str
    backend: str
    nnodes: int
    nproc_per_node: int
    node_rank: int
    world_size: int
    global_batch_size: int
    rendezvous: dict
    dtype: str
    runtime_profile: str = "legacy_qwen3_vl"

    def as_dict(self) -> dict:
        result = asdict(self)
        result.update({"schema_version": 1, "command": shlex.join(self.argv),
                       "environment_overrides": {"PYTHON_EXEC": self.python_executable},
                       "environment_removals": ["PET_* (torchrun option overrides)"],
                       "max_restarts": 0})
        return result


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_master(address: str) -> str:
    if not isinstance(address, str) or not address or address != address.strip():
        raise ValueError("Multi-node --master-addr must be a hostname or IP address")
    if address.lower().rstrip(".") == "localhost" or address.lower().rstrip(".").endswith(".localhost"):
        raise ValueError("Multi-node --master-addr cannot be localhost/loopback")
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        if re.fullmatch(r"[0-9.]+", address) or len(address) > 253 or not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                                        for label in address.rstrip(".").split(".")):
            raise ValueError("--master-addr must be a hostname or bare IP, without scheme, port or whitespace")
    else:
        if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
            raise ValueError("Multi-node --master-addr must identify a reachable node, not loopback/wildcard/multicast")
        if getattr(parsed, "ipv4_mapped", None) is not None:
            _validate_master(str(parsed.ipv4_mapped))
    return address


def build_plan(config_path: str | Path, *, nnodes: int, nproc_per_node: int,
               node_rank: int | None = None, master_addr: str | None = None,
               master_port: int | None = None, python_executable: str | None = None,
               environ: Mapping[str, str] | None = None) -> LaunchPlan:
    """Validate config/topology and construct argv without importing Torch or creating files."""
    from .training.config import load_config

    _positive_int(nnodes, "nnodes")
    _positive_int(nproc_per_node, "nproc_per_node")
    environment = os.environ if environ is None else environ
    if any(name in environment for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise ValueError("Already inside a distributed worker environment; run this launcher once per node, not under torchrun")
    config = load_config(config_path)
    world_size = nnodes * nproc_per_node
    expected = config.get("expected_world_size", 1)
    if expected != world_size:
        raise ValueError(f"Topology world_size={world_size} differs from config expected_world_size={expected}; "
                         "set topology and gradient accumulation deliberately")
    if node_rank is not None and (isinstance(node_rank, bool) or not isinstance(node_rank, int)
                                  or not 0 <= node_rank < nnodes):
        raise ValueError("node_rank must be an integer in [0, nnodes)")
    # Preserve the venv executable path: Path.resolve() would follow its Python symlink.
    python = os.path.abspath(python_executable or sys.executable)
    argv = [python, "-m", "torch.distributed.run", "--nnodes", str(nnodes),
            "--nproc-per-node", str(nproc_per_node), "--max-restarts", "0"]
    if nnodes == 1:
        if master_addr is not None or master_port is not None:
            raise ValueError("Single-node launch uses --standalone and an automatic free port; omit master address/port")
        argv += ["--standalone"]
        rendezvous = {"mode": "standalone", "port": "automatic"}
        actual_node_rank = 0
    else:
        if node_rank is None or master_addr is None:
            raise ValueError("Multi-node launch requires explicit --node-rank and --master-addr on every node")
        master_addr = _validate_master(master_addr)
        port = 29500 if master_port is None else master_port
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("master_port must be an integer between 1 and 65535")
        actual_node_rank = node_rank
        argv += ["--rdzv-backend", "static", "--node-rank", str(node_rank),
                 "--master-addr", f"[{master_addr}]" if ":" in master_addr else master_addr, "--master-port", str(port),
                 "--rdzv-conf", f"timeout={config.get('distributed_timeout_seconds', 300)}"]
        rendezvous = {"mode": "static", "master_addr": master_addr, "master_port": port}
    # Persist each worker's full traceback and preparation heartbeats even when
    # the scheduler truncates its combined stdout/stderr stream.
    argv += ["--log-dir", str(Path(config["config_path"]).parent / "torchrun"),
             "--redirects", "3", "--tee", "3",
             "--module", "optics_agent_sft", "train", "--config", config["config_path"]]
    return LaunchPlan(tuple(argv), config["config_path"], config["config_sha256"], python,
                      config.get("distributed_backend", "ddp"), nnodes, nproc_per_node,
                      actual_node_rank, world_size,
                      world_size * config.get("per_device_train_batch_size", 1)
                      * config.get("gradient_accumulation_steps", 8), rendezvous,
                      config.get("dtype", "bf16"), config.get('runtime_profile','legacy_qwen3_vl'))


_PREFLIGHT_CODE = r'''
import json, sys, faulthandler, signal, traceback
faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
def event(**fields):
    print("OPTICS_SFT_PREFLIGHT " + json.dumps(fields), flush=True)
def stage(name):
    event(event="stage", stage=name)
event(event="runtime_started", python=sys.executable)
try:
    cublas_report = {"required": False, "reason": "legacy runtime profile"}
    gemm_report = {"required": False, "reason": "legacy runtime profile"}
    if sys.argv[4] == "qwen_modern":
        from optics_agent_sft.cublas_runtime import verify_loaded_cublas
        stage("verify paired cuBLAS/cuBLASLt libraries")
        cublas_report = verify_loaded_cublas()
        event(event="cublas_runtime", report=cublas_report)
    compiler_report = {"required": False, "reason": "legacy runtime profile"}
    if sys.argv[4] == "qwen_modern":
        from optics_agent_sft.training.config import load_config
        from optics_agent_sft.compiler_preflight import run_compiler_preflight
        stage("model compiler metadata")
        compiler_report = run_compiler_preflight(load_config(sys.argv[5]), stage=stage)
    stage("import torch")
    import torch
    stage("package versions")
    import importlib.metadata
    from packaging.version import Version
    nproc = int(sys.argv[1])
    backend, dtype = sys.argv[2:4]
    packages = ["torch", "transformers", "accelerate", "peft", "Pillow"]
    if backend == "deepspeed_zero2":
        packages.append("deepspeed")
        if sys.platform != "linux":
            raise RuntimeError("DeepSpeed execution is supported only on Linux CUDA hosts")
    versions = {name: importlib.metadata.version(name) for name in packages}
    from optics_agent_sft.training.architectures import required_versions
    required=required_versions({'runtime_profile':sys.argv[4],'distributed_backend':backend})
    for name, expected in required.items():
        if versions[name] != expected:
            raise RuntimeError(f"This runtime profile requires {name}=={expected}; found {versions[name]}")
    if Version(versions["torch"]) < Version("2.6"):
        raise RuntimeError("PyTorch >= 2.6 is required, including safe Trainer checkpoint loading")
    if sys.argv[4] == "qwen_modern" and Version(versions["torch"].split("+")[0]) != Version("2.10.0"):
        raise RuntimeError(f"qwen_modern requires torch==2.10.0; found {versions['torch']}")
    stage("CUDA availability")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; --dry-run prints a plan without requiring GPUs")
    count = torch.cuda.device_count()
    if count < nproc:
        raise RuntimeError(f"Requested {nproc} local workers but only {count} CUDA devices are visible")
    stage("NCCL support")
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise RuntimeError("A PyTorch build with NCCL distributed support is required")
    devices = []
    for index in range(nproc):
        stage(f"CUDA device {index}")
        with torch.cuda.device(index):
            if dtype == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"CUDA device {index} does not support bf16; choose fp16 explicitly")
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name, "total_memory_bytes": props.total_memory})
    if sys.argv[4] == "qwen_modern":
        # Check ordinary CUDA arithmetic before importing the transformer/FLA
        # stack, whose shared-disk imports can take several minutes.
        cublas_report = verify_loaded_cublas()
        event(event="cublas_runtime_after_torch", report=cublas_report)
        from optics_agent_sft.rotary_preflight import run_gemm_preflight
        gemm_report = run_gemm_preflight(dtype=dtype, device="cuda:0", stage=stage)
    kernel_report = {"required": False, "reason": "legacy runtime profile"}
    if sys.argv[4] == "qwen_modern":
        from optics_agent_sft.training.config import load_config
        from optics_agent_sft.kernel_preflight import run_model_kernel_preflight
        stage("model kernel metadata")
        kernel_report = run_model_kernel_preflight(load_config(sys.argv[5]), device=0, stage=stage)
    event(event="result", report={"ok": True, "versions": versions, "cuda_version": torch.version.cuda,
                                  "visible_cuda_devices": count, "used_devices": devices,
                                  "compiler_preflight": compiler_report,
                                  "cublas_runtime": cublas_report, "gemm_preflight": gemm_report,
                                  "model_kernel_preflight": kernel_report})
except Exception as error:
    traceback.print_exc()
    event(event="result", report={"ok": False, "error": str(error), "error_type": type(error).__name__})
    sys.exit(2)
'''


def execution_environment(plan: LaunchPlan, environ: Mapping[str, str] | None = None) -> dict:
    # torchrun's env/check_env argparse actions can silently override topology,
    # Python/module mode and rank behavior. Preserve device/network settings;
    # modern cuBLAS library selection is pinned separately below.
    if (plan.runtime_profile == "qwen_modern"
            and Path(plan.python_executable).absolute().parent.resolve()
            != Path(sys.executable).absolute().parent.resolve()):
        # Resolving the executable itself is insufficient: two venvs may use
        # symlinks to the same Python binary but different package metadata.
        raise ValueError("Modern SFT launcher and workers must use the same Python environment; "
                         "run optics_agent_sft.launch with the target environment's Python")
    environment = {key: value for key, value in (os.environ if environ is None else environ).items()
                   if not key.startswith("PET_")}
    environment["PYTHON_EXEC"] = plan.python_executable
    from .runtime_environment import configure_training_environment
    environment = configure_training_environment(environment, runtime_profile=plan.runtime_profile)
    if plan.runtime_profile == "qwen_modern":
        # Applies before exec to the probe AND torchrun (which imports torch
        # before workers). Workers inherit the same paired library preload.
        from .cublas_runtime import prepare_cublas_environment
        environment = prepare_cublas_environment(environment)
    return environment


def local_preflight(plan: LaunchPlan, *, timeout_seconds=1200, progress_interval=10, log_path=None) -> dict:
    """Check this node only; no model weights, training output, or peer checks."""
    from .preflight import run_probe

    if hashlib.sha256(Path(plan.config_path).read_bytes()).hexdigest() != plan.config_sha256:
        raise ValueError("Config changed after the launch plan was constructed")
    if log_path is None:
        log_path = Path(plan.config_path).parent / f"local_preflight_rank{plan.node_rank}.log"
    return run_probe([plan.python_executable, "-u", "-c", _PREFLIGHT_CODE,
                      str(plan.nproc_per_node), plan.backend, plan.dtype, plan.runtime_profile, plan.config_path],
                     environment=execution_environment(plan), log_path=log_path,
                     timeout_seconds=timeout_seconds, progress_interval=progress_interval)


def supervise(argv: Sequence[str], *, environment: Mapping[str, str], shutdown_grace_seconds: float = 30) -> int:
    """Forward interrupt/termination to the torchrun process group and preserve failure status."""
    if shutdown_grace_seconds <= 0:
        raise ValueError("shutdown_grace_seconds must be positive")
    child = None
    received_signal = None
    deadline = None
    previous_handlers = {}

    def forward(signum: int) -> None:
        if child is None:
            return
        try:
            if os.name == "posix":
                os.killpg(child.pid, signum)
            else:
                child.send_signal(signum)
        except ProcessLookupError:
            pass

    def on_signal(signum: int, _frame) -> None:
        nonlocal received_signal, deadline
        if received_signal is None:
            received_signal = signum
            deadline = time.monotonic() + shutdown_grace_seconds
            forward(signum)
        else:
            forward(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, *([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])):
            previous_handlers[signum] = signal.signal(signum, on_signal)
        if received_signal is not None:
            return 128 + received_signal
        child = subprocess.Popen(list(argv), env=dict(environment), shell=False, start_new_session=os.name == "posix")
        if received_signal is not None:
            forward(received_signal)
        while True:
            try:
                code = child.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    forward(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
            except KeyboardInterrupt:
                on_signal(signal.SIGINT, None)
        return (128 + received_signal) if received_signal is not None else (128 - code if code < 0 else code)
    finally:
        if child is not None and (received_signal is not None or child.poll() is None):
            # Kill any remaining members even if the group leader already exited.
            forward(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
            child.wait()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: Sequence[str] | None = None) -> int:
    from .preflight import PreflightInterrupted, validate_timing

    parser = argparse.ArgumentParser(description="Launch fixed-size CUDA SFT with torchrun, once per node")
    parser.add_argument("--config", required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print validated topology/argv only; no GPU checks or subprocesses")
    parser.add_argument("--preflight-timeout-seconds", type=float, default=1200,
                        help="Bound local environment/CUDA checks (default: 1200 seconds)")
    parser.add_argument("--preflight-progress-interval", type=float, default=10,
                        help="Seconds between progress updates during a blocked check (default: 10)")
    parser.add_argument("--preflight-log", help="Probe log path; defaults beside config to local_preflight_rankN.log")
    parser.add_argument("--preflight-only", action="store_true", help="Check local environment/CUDA and exit before torchrun")
    args = parser.parse_args(argv)
    try:
        validate_timing(args.preflight_timeout_seconds, args.preflight_progress_interval)
        plan = build_plan(args.config, nnodes=args.nnodes, nproc_per_node=args.nproc_per_node,
                          node_rank=args.node_rank, master_addr=args.master_addr, master_port=args.master_port)
        report = plan.as_dict()
        report["dry_run"] = args.dry_run
        if args.dry_run:
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            return 0
        report["local_preflight"] = local_preflight(plan, timeout_seconds=args.preflight_timeout_seconds,
            progress_interval=args.preflight_progress_interval, log_path=args.preflight_log)
        report["preflight_only"] = args.preflight_only
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.preflight_only:
            return 0
        return supervise(plan.argv, environment=execution_environment(plan))
    except PreflightInterrupted as error:
        return 128 + error.signum
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Launch failed: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
