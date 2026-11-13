"""Training lifecycle, from admission to a verified adapter artifact.

Read this module first for the execution order. Details live in provenance
(data/model identities), modeling (Qwen/LoRA construction), trainer_integration
(loss/preflight/checkpoint hooks), and checkpoints (disk state and output leases).
Imports and CPU dry-runs never load GPU training dependencies.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import (RUN_MANIFEST, acquire_output, atomic_json, file_inventory,
                          release_output, validate_checkpoint, verify_output_visibility)
from .config import load_config, local_phase_options, semantic_config
from .distributed import Coordinator, Topology, bootstrap_cuda, configure_accelerate_environment
from .modeling import (dependency_versions, load_dependencies, load_language_adapter,
                       load_processor, resolve_model_snapshot)
from .provenance import admit_training_data, dataset_identity, initial_manifest, model_identity
from .trainer_integration import check_device, prepare_trainer, training_arguments

# Preserve the original import surface used by existing tests and CPU smoke tools.
# These are direct aliases, not another implementation layer.
from .modeling import load_dependencies as _load_dependencies, resolve_model_snapshot as _resolve_model
from .provenance import admit_training_data as _admit, initial_manifest as _manifest
from .trainer_integration import check_device as _check_device, check_trainable as _check_trainable, trainer_classes as _trainer_classes


def _initialize_distributed_run(config_path: str | Path) -> tuple[dict[str, Any], Coordinator]:
    """Report local config errors collectively before any worker loads a model."""
    config, config_error = None, None
    try:
        config = load_config(config_path)
    except Exception as error:
        config_error = error

    topology = Topology.from_environment()
    coordinator = bootstrap_cuda(topology, timeout_seconds=(config or {}).get("distributed_timeout_seconds", 300))

    def validate_config() -> dict[str, Any]:
        if config_error:
            raise config_error
        if topology.world_size != config["expected_world_size"]:
            raise ValueError(f"WORLD_SIZE={topology.world_size} differs from expected_world_size={config['expected_world_size']}; use torchrun")
        return config

    config = coordinator.phase("validate training config", validate_config)
    coordinator.agree("training configuration", semantic_config(config))
    # Logging/save policy is absent from the checkpoint's semantic identity.
    # Within one run every worker must nevertheless enter
    # Trainer's logging and checkpoint collectives at the same steps.
    coordinator.agree("logging and checkpoint intervals", {
        key: config[key] for key in ("logging_steps", "save_steps", "save_strategy")
    })
    coordinator.agree("local preparation timing", {
        key: config[key] for key in ("preparation_timeout_seconds", "preparation_progress_interval_seconds")
    })
    coordinator.phase("apply explicit Trainer precision/backend environment", lambda: configure_accelerate_environment(config))
    return config, coordinator


def _admit_node_dataset(config, coordinator, *, allow_demo=False):
    """Hash/re-derive approved data once per node, then share the verified rows.

    Local ranks must use the same filesystem paths. Other nodes independently
    validate their own snapshots; their byte identities are compared below.
    Image-aware prefix validation remains sharded across all training ranks.
    """
    for name in ("dataset_dir", "runs", "reserved_manifests"):
        value = config.get(name)
        paths = value if isinstance(value, list) else [value] if value else []
        resolved = coordinator.phase(
            f"resolve node-local {name}", lambda: [str(Path(path).resolve()) for path in paths],
        )
        coordinator.agree_node_path(name, resolved)
    local = coordinator.local_phase(
        "admit approved dataset on node leaders",
        lambda: admit_training_data(config, dry_run=False, allow_demo=allow_demo)
        if coordinator.topology.local_rank == 0 else None,
        **local_phase_options(config),
    )
    return coordinator.phase("share verified dataset within each node", lambda: coordinator.broadcast_node(local))


def run_training(config_path: str | Path, *, dry_run: bool = False, allow_demo: bool = False) -> dict[str, Any]:
    """Train only approved agent decisions, or perform dependency-free admission.

    GPU execution follows five ordered stages below. Every preparation or save
    phase gathers ordinary rank-local failures; hard process/collective failures
    remain the responsibility of torchrun and its bounded process-group timeout.
    """
    if allow_demo and not dry_run:
        raise ValueError("Demo trajectories are only allowed in dry-run, never training")
    if dry_run:
        config = load_config(config_path)
        admission, _, _ = admit_training_data(config, dry_run=True, allow_demo=allow_demo)
        return initial_manifest(config, admission, dry_run=True, topology=Topology())

    # 1. Agree on topology and approved data before importing/loading model code.
    config, coordinator = _initialize_distributed_run(config_path)
    topology = coordinator.topology
    admission, training, validation = _admit_node_dataset(config, coordinator, allow_demo=allow_demo)
    data_id = coordinator.local_phase("fingerprint ordered data", lambda: dataset_identity(admission),
                                      **local_phase_options(config))
    coordinator.agree("dataset and human approval provenance", data_id)
    dependencies = coordinator.phase("load training dependencies", lambda: load_dependencies(config))
    versions = coordinator.phase("read training dependency versions", lambda: dependency_versions(config))
    coordinator.agree("training dependency versions", versions)

    # 2. Initialize Accelerator/DeepSpeed device state before constructing Qwen.
    args = coordinator.phase("initialize Trainer distributed device", lambda: dependencies.training_arguments(
        **training_arguments(config, has_validation=bool(validation), local_rank=topology.local_rank),
    ))
    coordinator.phase("validate Trainer device", lambda: check_device(args, topology))
    model_path, model_id = resolve_model_snapshot(config, coordinator)
    identity = {
        "schema_version": 1, "data": data_id, "model": model_id,
        "semantic_config": semantic_config(config), "versions": versions,
        "world_size": topology.world_size, "local_world_size": topology.local_world_size,
        "loss_version": "global-supervised-token-causal-v2-selective-qwen-logits",
    }
    coordinator.agree("complete resume identity", identity)

    # 3. Validate full-state resume and acquire one shared, exclusively owned output.
    checkpoint = config["resume_from_checkpoint"]
    coordinator.agree("fresh versus resume mode", bool(checkpoint))
    if checkpoint:
        coordinator.agree_node_path("resume checkpoint", Path(checkpoint).resolve())
        restored = coordinator.local_phase("validate full resume checkpoint", lambda: validate_checkpoint(
            checkpoint, identity, backend=config["distributed_backend"], world_size=topology.world_size,
            dtype=config["dtype"], verify_hashes=topology.local_rank == 0,
        ), **local_phase_options(config))
        coordinator.agree("resume checkpoint", restored)
    # This can fail during provenance projection; do it collectively before
    # acquiring an output lease so an ordinary error cannot leak a stale lock.
    manifest = coordinator.local_phase("prepare initial run manifest", lambda: initial_manifest(
        config, admission, dry_run=False, topology=topology,
    ), **local_phase_options(config))
    output = Path(config["output_dir"])
    lease = coordinator.phase("acquire output directory", lambda: acquire_output(output, identity=identity, resume=checkpoint)
                              if topology.rank == 0 else None)
    lease = coordinator.gather(lease)[0]
    manifest.update({"identity": identity, "status": "preparing", "versions": versions,
                     "resume_from_checkpoint": checkpoint, "run_id": lease["run_id"]})
    try:
        coordinator.phase("verify shared output storage", lambda: verify_output_visibility(output, lease))
        coordinator.phase("write initial run manifest", lambda: atomic_json(output / RUN_MANIFEST, manifest) if topology.rank == 0 else None)

        # 4. Build frozen Qwen + language LoRA, and validate every exact image-aware prefix.
        dependencies.set_seed(config["seed"])
        processor = coordinator.phase("load Qwen processor", lambda: load_processor(model_path, config, dependencies))
        model, targets = coordinator.phase("load frozen base and language LoRA", lambda: load_language_adapter(model_path, config, dependencies))
        prepared = prepare_trainer(
            config=config, dependencies=dependencies, coordinator=coordinator, identity=identity,
            args=args, model=model, processor=processor, targets=targets, admission=admission,
            training=training, validation=validation,
        )
        trainer = prepared.trainer
        manifest.update({"status": "training", "model_type": model.config.model_type, "trainable_parameters": prepared.trainable_parameters,
                         "lora_target_modules": targets, "validated_supervised_tokens": prepared.supervised_tokens,
                         "prefix_validation_shards": prepared.prefix_validation_shards})
        coordinator.phase("write training manifest", lambda: atomic_json(output / RUN_MANIFEST, manifest) if topology.rank == 0 else None)

        # 5. Trainer restores optimizer/scheduler/scaler/RNG/data position; all ranks save.
        result = trainer.train(resume_from_checkpoint=checkpoint)
        coordinator.phase("save final adapter on participating ranks", lambda: trainer.save_model(str(output / "adapter")))

        def finalize() -> None:
            if topology.rank != 0:
                return
            processor.save_pretrained(str(output / "adapter"))
            manifest.update({"status": "completed", "metrics": result.metrics, "global_step": trainer.state.global_step,
                             "completed_at": datetime.now(timezone.utc).isoformat(),
                             "adapter_files": file_inventory(output / "adapter")})
            atomic_json(output / RUN_MANIFEST, manifest)

        coordinator.phase("publish final adapter manifest", finalize)
        return coordinator.gather(manifest)[0]
    except BaseException as error:
        if topology.rank == 0:
            manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            try:
                atomic_json(output / RUN_MANIFEST, manifest)
            except OSError:
                pass
        raise
    finally:
        if topology.rank == 0:
            release_output(output, lease)
