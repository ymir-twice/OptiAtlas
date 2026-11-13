"""CPU-only admission for the narrowly pinned vLLM FileStore backport.

Do not import vLLM/torch here: this check runs before Ray or GPU preparation.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path


EXECUTOR = "optics_agent_rl.vllm_executor.FileStoreMultiprocExecutor"


def _method(source: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    return method
    raise ValueError(f"Missing {class_name}.{method_name}")


def _normalized(method: ast.FunctionDef) -> str:
    method = copy.deepcopy(method)
    method.name = "_init_executor"
    changes = 0
    for node in ast.walk(method):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "distributed_init_method"):
            node.value = ast.Constant(value="<rendezvous-only-backport>")
            changes += 1
    if changes != 1:
        raise ValueError(f"Expected one rendezvous assignment, found {changes}")
    return ast.dump(method, include_attributes=False)


def validate_pinned_source(path: str | Path, *, implementation_path=None) -> dict:
    """Reject API/source drift instead of silently running an incomplete copy."""
    source_path = Path(path)
    implementation_path = Path(implementation_path or Path(__file__).with_name("vllm_executor.py"))
    try:
        original = source_path.read_text(encoding="utf-8")
        implementation = implementation_path.read_text(encoding="utf-8")
        upstream = _method(original, "MultiprocExecutor", "_init_executor")
        backport = _method(implementation, "FileStoreMultiprocExecutor", "_init_file_executor")
        if _normalized(upstream) != _normalized(backport):
            raise ValueError("Executor initialization differs beyond the rendezvous assignment")
    except (OSError, SyntaxError, ValueError) as error:
        raise RuntimeError(
            f"FileStore executor is incompatible with {source_path}: {error}. "
            "Use the project's pinned vLLM 0.19.1 runtime or update and test the backport."
        ) from error
    return {"backend": EXECUTOR, "source": str(source_path),
            "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "implementation_sha256": hashlib.sha256(implementation.encode()).hexdigest(),
            "status": "compatible"}


def validate_selected_runtime(python: str | Path) -> dict:
    """Locate the selected overlay's actual package without importing it."""
    prefix = Path(python).absolute().parent.parent  # Keep the venv, not a symlink target.
    sites = sorted((prefix / "lib").glob("python*/site-packages"))
    record = prefix / "base-runtime.json"
    if record.is_file():
        sites.append(Path(json.loads(record.read_text(encoding="utf-8"))["site_packages"]))
    for site in sites:
        source = site / "vllm/v1/executor/multiproc_executor.py"
        if not source.is_file():
            continue
        versions = []
        for metadata in site.glob("vllm-*.dist-info/METADATA"):
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version: "):
                    versions.append(line.partition(": ")[2])
                    break
        if versions != ["0.19.1"]:
            raise RuntimeError(f"FileStore backport requires pinned vLLM 0.19.1; found {versions} in {site}")
        return {**validate_pinned_source(source), "version": versions[0]}
    raise RuntimeError(f"Cannot locate vLLM for {python}; prepare scripts/setup_agent_v3_rl_env.sh first")
