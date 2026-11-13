"""Transactional checkpoints for the Draw-Align OPSD integration.

Keep the pinned vendor untouched. Completion metadata covers both continuation
state and inference exports; older checkpoints are accepted only after checking
their required state files and explicit trainer position.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid


logger = logging.getLogger(__name__)
_STEP_RE = re.compile(r"checkpoint-(\d+)$")
_PROVENANCE = "upstream_provenance.json"
_IDENTITY_KEYS = (
    "data_sha256", "model_config", "finetune_config", "world_size",
    "effective_model_config", "reward_config",
)
_LEGACY_UPSTREAM_KEYS = ("sample", "train", "opsd", "reward_fn", "resolution")
_ADDITIONAL_UPSTREAM_KEYS = (
    "beta", "decay_type", "per_prompt_stat_tracking", "mixed_precision", "allow_tf32", "seed",
)
_UPSTREAM_KEYS = _LEGACY_UPSTREAM_KEYS + _ADDITIONAL_UPSTREAM_KEYS


def _json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read checkpoint metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata must be an object: {path}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(root):
    state = _json(root / "trainer_state.json")
    for key in ("global_step", "epoch_completed"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError(f"Checkpoint has invalid {key}: {root}")
    match = _STEP_RE.fullmatch(root.name)
    if match and int(match.group(1)) != state["global_step"]:
        raise ValueError(f"Checkpoint directory and trainer global_step disagree: {root}")
    return state


def _files(root):
    return {str(path.relative_to(root)): path for path in root.rglob("*")
            if path.is_file() and path != root / "complete.json"}


def _required_files(root, *, require_ema, require_scaler, native_policy):
    required = {"optimizer.pt", "trainer_state.json"}
    if native_policy:
        required.add("policy.pt")
    if require_ema:
        required.add("ema.pt")
    if require_scaler:
        required.add("scaler.pt")
    for name in sorted(required):
        path = root / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Checkpoint is missing required state {name}: {root}")
    exports = [path for path in (root / "lora").rglob("*") if path.is_file() and path.stat().st_size]
    if not exports and not (not native_policy and (root / "transformer_model.pt").is_file()):
        raise ValueError(f"Checkpoint is missing its model export: {root}")
    return required


def validate_checkpoint(path, *, require_ema=False, require_scaler=False, native_policy=True,
                        _allow_staging=False):
    """Check completion/integrity and return the explicit saved trainer position.

    For legacy directories, deserialize only the required continuation files on
    the meta device, so truncated state cannot be selected as a valid resume.
    """
    root = Path(path)
    if not root.is_dir() or (not _allow_staging and ".incomplete." in root.name):
        raise ValueError(f"Checkpoint is incomplete or missing: {root}")
    complete_path = root / "complete.json"
    manifest = _json(complete_path) if complete_path.exists() else None
    if manifest is not None:
        if manifest.get("format") != 1 or manifest.get("native_policy") is not native_policy:
            raise ValueError(f"Unsupported checkpoint format or policy type: {root}")
        for key in ("require_ema", "require_scaler"):
            if type(manifest.get(key)) is not bool:
                raise ValueError(f"Invalid checkpoint completion metadata {key}: {root}")
        require_ema = require_ema or manifest["require_ema"]
        require_scaler = require_scaler or manifest["require_scaler"]
    required = _required_files(root, require_ema=require_ema, require_scaler=require_scaler,
                               native_policy=native_policy)
    state = _state(root)
    if manifest is not None:
        if any(manifest.get(key) != state[key] for key in ("global_step", "epoch_completed")):
            raise ValueError(f"Checkpoint completion metadata disagrees with trainer state: {root}")
        hashes, files = manifest.get("files_sha256"), _files(root)
        if not isinstance(hashes, dict) or set(hashes) != set(files):
            raise ValueError(f"Checkpoint completion manifest does not cover its files: {root}")
        for name, file in files.items():
            if _sha256(file) != hashes[name]:
                raise ValueError(f"Checkpoint checksum mismatch for {name}: {root}")
    else:
        # Old public checkpoints have no marker or checksums. Do not accept a
        # new, self-contained snapshot after its completion marker was removed.
        if (root / _PROVENANCE).exists():
            raise ValueError(f"Checkpoint is missing its completion manifest: {root}")
        import torch
        for name in sorted(required - {"trainer_state.json"}):
            try:
                value = torch.load(root / name, map_location="meta", weights_only=True)
            except Exception as error:
                raise ValueError(f"Unreadable legacy checkpoint state {name}: {root}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Invalid legacy checkpoint state {name}: {root}")
    return state


def resolve_resume_checkpoint(path, *, require_ema=False, require_scaler=False, native_policy=True):
    """Resolve an exact checkpoint or the newest valid checkpoint in a run."""
    if not path:
        return ""
    root = Path(path).expanduser().resolve()
    options = dict(require_ema=require_ema, require_scaler=require_scaler, native_policy=native_policy)
    if _STEP_RE.fullmatch(root.name) or (root / "complete.json").exists() or (root / "trainer_state.json").exists():
        validate_checkpoint(root, **options)
        return str(root)
    candidates = set()
    for parent in (root, root / "checkpoints", root / "save" / "checkpoints"):
        if parent.is_dir():
            candidates.update(child for child in parent.glob("checkpoint-*")
                              if child.is_dir() and _STEP_RE.fullmatch(child.name))
    for candidate in sorted(candidates, key=lambda item: int(_STEP_RE.fullmatch(item.name).group(1)), reverse=True):
        try:
            validate_checkpoint(candidate, **options)
        except (OSError, ValueError) as error:
            logger.warning("Skipping incomplete or invalid checkpoint %s: %s", candidate, error)
            continue
        return str(candidate)
    raise FileNotFoundError(f"No complete checkpoint found under {root}")


def load_run_provenance(path):
    """Load old run metadata, supplementing fields from its full saved config."""
    path = Path(path)
    if path.is_dir():
        path /= _PROVENANCE
    previous = _json(path)
    upstream = previous.get("upstream_config", {})
    missing = [key for key in _ADDITIONAL_UPSTREAM_KEYS if key not in upstream]
    if missing:
        sidecar = path.with_name("upstream_config.json")
        if not sidecar.is_file():
            raise ValueError(f"Saved run provenance lacks upstream {missing}; source upstream_config.json is required: {path.parent}")
        complete_config = _json(sidecar)
        for key in missing:
            if key not in complete_config:
                raise ValueError(f"Source upstream_config.json is missing {key}; cannot establish resume identity")
            upstream[key] = complete_config[key]
        previous["upstream_config"] = upstream
    return previous


def load_checkpoint_provenance(path):
    """Prefer the checkpoint snapshot; support the old run/checkpoints layout."""
    root = Path(path).resolve()
    snapshot = root / _PROVENANCE
    if snapshot.is_file():
        return _json(snapshot)
    # Only known source layouts are searched, never the destination directory
    # or an arbitrary ancestor that may belong to a different experiment.
    candidates = [root.parent / _PROVENANCE]
    if root.parent.name == "checkpoints":
        candidates.append(root.parent.parent / _PROVENANCE)
        if root.parent.parent.name == "save":
            candidates.append(root.parent.parent.parent / _PROVENANCE)
    for candidate in candidates:
        if candidate.is_file():
            return load_run_provenance(candidate)
    raise ValueError(f"Resume requires the checkpoint's source upstream_provenance.json: {root}")


def validate_saved_initial_reference(path, payload):
    """Read-only verification of the opt-in v2 initial-anchor checkpoint.

    The checksum proves the saved frozen-bank identity; it does not independently
    reload an 8B base model. Training also checks this identity against the exact
    run-initial bank on save and resume. Historical v1 files remain unchanged.
    """
    from .config import reference_velocity_weight
    version = payload.get("version")
    if version not in (1, 2):
        raise ValueError("Unsupported policy bank version")
    path = Path(path)
    metadata = path / _PROVENANCE
    upstream = _json(metadata).get("upstream_config", {}) if metadata.is_file() else {}
    weight = reference_velocity_weight(upstream.get("opsd", {}))
    if version == 1:
        if weight or any(key in payload for key in ("initial_reference", "initial_reference_sha256", "initial_reference_source")):
            raise ValueError("Legacy policy version cannot declare an enabled initial reference")
        return None
    if not metadata.is_file() or weight <= 0:
        raise ValueError("Initial-reference v2 policy requires matching positive-weight checkpoint provenance")
    import torch
    from .bridge import _bank_sha256
    names, initial, student = payload.get("names", []), payload.get("initial_reference", []), payload.get("default", [])
    if (not names or not all(isinstance(name, str) for name in names) or len(set(names)) != len(names)
            or len(initial) != len(names) or len(student) != len(names)
            or payload.get("initial_reference_source") != "initial"):
        raise ValueError("Invalid saved initial-reference bank metadata")
    for name, value, trainable in zip(names, initial, student, strict=True):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.shape != trainable.shape or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid saved initial-reference FP32 tensor: {name}")
    identity = _bank_sha256(names, initial)
    if identity != payload.get("initial_reference_sha256"):
        raise ValueError("Saved initial-reference bank checksum differs")
    return {"source": "initial", "weight": weight, "bank_sha256": identity,
            "tensors": len(names), "elements": sum(value.numel() for value in initial),
            "dtype": "float32", "finite": True, "saved_identity_verified": True,
            "initial_model_reloaded": False}


def validate_resume_provenance(previous, current):
    """Compare training identity while allowing longer runs and new outputs."""
    for key in _IDENTITY_KEYS:
        if key not in previous:
            raise ValueError(f"Saved run provenance is missing {key}")
        if previous[key] != current[key]:
            if key == "data_sha256":
                raise ValueError("Training/validation manifest changed since the saved upstream run")
            raise ValueError(f"Resume {key} differs from the saved run")
    old_upstream = previous.get("upstream_config", {})
    for key in _UPSTREAM_KEYS:
        if key not in old_upstream:
            raise ValueError(f"Saved run provenance is missing upstream {key}")
        old_value, new_value = old_upstream[key], current["upstream_config"][key]
        if key == "train":
            from .optimizer import canonical_train_identity
            old_value = canonical_train_identity(old_value)
            new_value = canonical_train_identity(new_value)
        if old_value != new_value:
            raise ValueError(f"Resume upstream {key} differs; start a new run for an ablation")


def provenance_for_save(config):
    """Metadata supplied by the public launcher, with a legacy launcher fallback."""
    value = getattr(config, "draw_align_provenance", None)
    if value is not None:
        return value.to_dict() if hasattr(value, "to_dict") else dict(value)
    path = Path(config.save_dir) / _PROVENANCE
    return load_run_provenance(path) if path.is_file() else None


def _same_complete_checkpoint(root, expected, options, provenance):
    # Validate a new-format save against its own declared requirements first.
    # A complete save made with different settings is a conflict, not a corrupt
    # directory that can be quarantined just because the new settings differ.
    native_policy = options["native_policy"]
    try:
        if (root / "complete.json").exists():
            manifest = _json(root / "complete.json")
            saved_native = manifest.get("native_policy")
            if type(saved_native) is bool:
                native_policy = saved_native
            saved = validate_checkpoint(root, native_policy=native_policy)
        else:
            saved = validate_checkpoint(root, **options)
    except (OSError, ValueError) as error:
        logger.warning("Existing checkpoint is invalid and will be preserved before replacement: %s: %s", root, error)
        return False
    if native_policy != options["native_policy"]:
        raise FileExistsError(f"Refusing to replace a checkpoint with a different policy type: {root}")
    _required_files(root, **options)
    if any(saved[key] != value for key, value in expected.items()):
        raise FileExistsError(f"Refusing to replace a checkpoint with conflicting trainer state: {root}")
    if provenance is not None:
        validate_resume_provenance(load_checkpoint_provenance(root), provenance)
    logger.info("Checkpoint already complete at the same training position: %s", root)
    return True


@contextmanager
def checkpoint_transaction(path, *, global_step, epoch_completed, require_ema=False,
                           require_scaler=False, native_policy=True, provenance=None):
    """Yield a staging directory, or None for an identical completed save.

    A complete old checkpoint is never replaced. An invalid old directory is
    preserved under a hidden name only after its replacement is ready.
    """
    root = Path(path)
    expected = {"global_step": int(global_step), "epoch_completed": int(epoch_completed)}
    options = dict(require_ema=require_ema, require_scaler=require_scaler, native_policy=native_policy)
    match = _STEP_RE.fullmatch(root.name)
    if match and int(match.group(1)) != expected["global_step"]:
        raise ValueError("Checkpoint directory and requested global_step disagree")
    if root.exists():
        if _same_complete_checkpoint(root, expected, options, provenance):
            yield None
            return
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.incomplete.", dir=root.parent))
    try:
        yield staging
        _required_files(staging, **options)
        state = _state(staging)
        if any(state[key] != value for key, value in expected.items()):
            raise ValueError("Saved trainer position differs from the checkpoint transaction")
        if provenance is not None:
            (staging / _PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n")
        manifest = {"format": 1, **expected, **options,
                    "files_sha256": {name: _sha256(file) for name, file in _files(staging).items()}}
        (staging / "complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
        validate_checkpoint(staging, **options, _allow_staging=True)
        quarantined = None
        if root.exists():
            # Another writer may have published a complete checkpoint while we
            # wrote the staging directory. Check it again before moving it.
            if _same_complete_checkpoint(root, expected, options, provenance):
                return
            quarantined = root.with_name(f".{root.name}.invalid.{uuid.uuid4().hex}")
            os.rename(root, quarantined)
            logger.warning("Preserved invalid checkpoint %s at %s", root, quarantined)
        # POSIX rename cannot overwrite a nonempty checkpoint directory. This
        # also prevents two writers from silently replacing each other's state.
        try:
            os.rename(staging, root)
        except BaseException:
            if quarantined is not None and not root.exists():
                try:
                    os.rename(quarantined, root)
                except OSError:
                    logger.warning("Could not restore invalid checkpoint to %s; its files remain at %s", root, quarantined)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
