"""The pinned Hugging Face Trainer boundary: arguments, preflight and saves.

Checkpoint subclasses are built only after lazy dependency loading. Native
Trainer writes remain authoritative for model/optimizer/scheduler/RNG state;
the override coordinates ranks and publishes a verified completion marker
before allowing checkpoint rotation.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, NamedTuple

from ..data import Admission
from .checkpoints import publish_checkpoint
from .distributed import Coordinator, Topology
from .modeling import TrainingDependencies
from .architectures import SUPPORTED_MODEL_TYPES

def training_arguments(config: dict[str, Any], *, has_validation: bool, local_rank: int) -> dict[str, Any]:
    """Plain kwargs, independently testable without CUDA or Transformers."""
    return dict(output_dir=config["output_dir"],
                per_device_train_batch_size=config["per_device_train_batch_size"], per_device_eval_batch_size=1,
                gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"],
                num_train_epochs=config["num_train_epochs"], max_steps=config["max_steps"],
                bf16=config["dtype"] == "bf16", fp16=config["dtype"] == "fp16",
                gradient_checkpointing=config["gradient_checkpointing"],
                gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=config["logging_steps"], save_steps=config["save_steps"], save_strategy=config["save_strategy"],
                save_total_limit=config["save_total_limit"], save_only_model=False, save_on_each_node=False,
                remove_unused_columns=False, report_to=config.get('report_to',[]),
                run_name=config.get('run_name'), seed=config["seed"], data_seed=config["seed"],
                dataloader_num_workers=config["dataloader_num_workers"],
                dataloader_persistent_workers=config["dataloader_persistent_workers"],
                dataloader_prefetch_factor=config["dataloader_prefetch_factor"],
                dataloader_pin_memory=True, label_names=["labels"],
                eval_strategy="epoch" if has_validation else "no", prediction_loss_only=True,
                average_tokens_across_devices=True, local_rank=local_rank,
                ddp_backend="nccl", ddp_timeout=config["distributed_timeout_seconds"],
                ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
                deepspeed=copy.deepcopy(config["deepspeed"]), optim=config.get('optim','adamw_torch'),
                warmup_ratio=config.get('warmup_ratio',0.0), weight_decay=config.get('weight_decay',0.0),
                max_grad_norm=config.get('max_grad_norm',1.0), lr_scheduler_type=config.get('lr_scheduler_type','linear'),
                ignore_data_skip=False, restore_callback_states_from_checkpoint=False)



def trainer_classes(Trainer: type, TrainerCallback: type, coordinator: Coordinator,
                    identity: dict[str, Any], config: dict[str, Any]) -> tuple[type, type]:
    """Build the two pinned Trainer extensions without importing Transformers early."""
    class FinalCheckpointCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self._initial_step = state.global_step
            self._first_step_started = None
            self._first_substeps = 0
            if coordinator.topology.rank == 0:
                coordinator.progress("trainer_loop_started", global_step=state.global_step,
                                     max_steps=state.max_steps)

        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == getattr(self, "_initial_step", None):
                self._first_step_started = time.monotonic()
                if coordinator.topology.rank == 0:
                    coordinator.progress("first_optimizer_step_started", global_step=state.global_step,
                                         gradient_accumulation_steps=config["gradient_accumulation_steps"])

        def on_substep_end(self, args, state, control, **kwargs):
            if (state.global_step == getattr(self, "_initial_step", None)
                    and self._first_step_started is not None):
                self._first_substeps += 1
                if coordinator.topology.rank == 0:
                    coordinator.progress("first_optimizer_step_accumulating",
                                         completed_microbatches=self._first_substeps,
                                         elapsed_seconds=round(time.monotonic() - self._first_step_started, 3))

        def on_step_end(self, args, state, control, **kwargs):
            if (state.global_step == getattr(self, "_initial_step", -2) + 1
                    and self._first_step_started is not None and coordinator.topology.rank == 0):
                coordinator.progress("first_optimizer_step_completed", global_step=state.global_step,
                                     elapsed_seconds=round(time.monotonic() - self._first_step_started, 3))
            # Epoch saving runs after its evaluation, including the final epoch.
            # Do not save it twice or publish final state before that evaluation.
            # A short step-based run must still leave full optimizer/RNG state.
            if state.global_step >= state.max_steps and getattr(args, "save_strategy", "steps") != "epoch":
                control.should_save = True
            return control

    class VerifiedCheckpointTrainer(Trainer):
        def _get_dataloader(self, *args, **kwargs):
            dataloader = super()._get_dataloader(*args, **kwargs)
            if self.args.dataloader_num_workers:
                # Trainer constructs loaders after CUDA/NCCL initialization and
                # processor preflight. Forking that state is unsafe. Accelerate
                # 1.10 wraps the PyTorch loader; change its actual worker context
                # before iteration, including the cached persistent eval loader.
                base = getattr(dataloader, "base_dataloader", dataloader)
                base.multiprocessing_context = "spawn"
            return dataloader

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Keep HF's normalization/accumulation contract while avoiding logits
            # for visual/history positions that cannot contribute to the target.
            model_type = getattr(getattr(self.model, "config", None), "model_type", None)
            if model_type in SUPPORTED_MODEL_TYPES and self.compute_loss_func is not None:
                from .loss import selective_qwen_forward
                model = selective_qwen_forward(model, inputs["labels"])
            return super().compute_loss(model, dict(inputs), return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """Evaluate a stable macro mean of per-decision token losses.

            Training uses a global token denominator. Evaluation instead needs
            one independent scalar per decision before Trainer gathers results
            and removes distributed sampler padding. HF 4.57.6's default
            cross-rank denominator makes that scalar depend on the other ranks.
            """
            if not prediction_loss_only or inputs.get("labels") is None:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            import torch

            inputs = self._prepare_inputs(inputs)
            if inputs["labels"].ndim != 2 or inputs["labels"].shape[0] != 1:
                raise ValueError("Macro decision eval_loss requires per_device_eval_batch_size=1")
            with torch.no_grad(), self.compute_loss_context_manager():
                loss, _ = self.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=None)
            return loss.detach().mean(), None, None

        def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
            if getattr(self, "_defer_checkpoint_rotation", False):
                return
            if hasattr(Trainer,'_rotate_checkpoints'):
                return super()._rotate_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
            # Transformers 5.12 moved rotation out of the Trainer method.
            from transformers.trainer_utils import rotate_checkpoints
            return rotate_checkpoints(output_dir=output_dir or self.args.output_dir,
                save_total_limit=self.args.save_total_limit,
                best_model_checkpoint=self.state.best_model_checkpoint,use_mtime=use_mtime)

        def _save(self, *args, **kwargs):
            # Record the true retention policy in training_args.bin even while
            # modern Trainer's unconditional rotation is temporarily disabled.
            if hasattr(self,'_checkpoint_save_limit'):
                deferred = self.args.save_total_limit
                self.args.save_total_limit = self._checkpoint_save_limit
                try:
                    return super()._save(*args,**kwargs)
                finally:
                    self.args.save_total_limit = deferred
            return super()._save(*args,**kwargs)

        def _save_checkpoint(self, model, trial):
            # Pinned Transformers API. All ranks write their native RNG/DS shards.
            # Never prune the last good checkpoint before its successor is complete.
            self._defer_checkpoint_rotation = True
            modern = not hasattr(Trainer,'_rotate_checkpoints')
            if modern:
                self._checkpoint_save_limit = self.args.save_total_limit
                self.args.save_total_limit = None
            try:
                coordinator.phase("write native full checkpoint", lambda: super(VerifiedCheckpointTrainer, self)._save_checkpoint(model, trial))
            finally:
                self._defer_checkpoint_rotation = False
                if modern:
                    self.args.save_total_limit = self._checkpoint_save_limit
                    del self._checkpoint_save_limit
            path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            coordinator.phase("publish complete checkpoint", lambda: publish_checkpoint(
                path, identity, backend=config["distributed_backend"], world_size=coordinator.topology.world_size,
                dtype=config["dtype"]) if coordinator.topology.rank == 0 else None)
            coordinator.phase("rotate only after checkpoint completion", lambda: self._rotate_checkpoints(
                use_mtime=True, output_dir=self.args.output_dir) if coordinator.topology.rank == 0 else None)
    return VerifiedCheckpointTrainer, FinalCheckpointCallback



def check_device(args: Any, topology: Topology) -> None:
    """Disallow topology drift and implicit single-process DataParallel."""
    if args.device.type != "cuda" or args.device.index != topology.local_rank or args.world_size != topology.world_size:
        raise ValueError("Trainer device/process state disagrees with torchrun topology")
    if args.n_gpu != 1:
        raise ValueError("Each worker must use exactly one CUDA GPU; use the torchrun launcher or expose only one device")



def check_trainable(count: int) -> None:
    if count <= 0:
        raise ValueError("Adapter has no trainable parameters")


def check_trainer_runtime(trainer: Any, config: dict[str, Any]) -> None:
    """Reject inherited Accelerator state that differs from the recorded config."""
    expected_precision = {"bf16": "bf16", "fp16": "fp16", "fp32": "no"}[config["dtype"]]
    if trainer.accelerator.mixed_precision != expected_precision:
        raise ValueError("Actual Accelerator mixed precision differs from configured dtype")
    expected_deepspeed = config["distributed_backend"] == "deepspeed_zero2"
    if bool(trainer.is_deepspeed_enabled) != expected_deepspeed or trainer.is_fsdp_enabled or getattr(trainer,'is_tp_enabled',False):
        raise ValueError("Actual Trainer backend differs from configured DDP/ZeRO-2")
    if not expected_deepspeed and trainer.accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Trainer owns gradient accumulation; Accelerator must use one step to avoid dividing loss twice")
    # Accelerate 1.10's N-D mesh is separate from the legacy TP flag above.
    # This runner has no mesh configuration; inherited state must not enable it.
    if getattr(trainer.accelerator, "parallelism_config", None) is not None:
        raise ValueError("Actual Accelerator parallelism_config is unsupported; use configured DDP/ZeRO-2")
    if expected_deepspeed:
        plugin = trainer.accelerator.state.deepspeed_plugin
        if plugin.hf_ds_config.config["zero_optimization"]["stage"] != 2:
            raise ValueError("Actual DeepSpeed configuration is not ZeRO-2")


class PreparedTrainer(NamedTuple):
    """Trainer plus the exact preflight measurements recorded in the run manifest."""

    trainer: Any
    trainable_parameters: int
    supervised_tokens: int
    prefix_validation_shards: list[int]


def prepare_trainer(*, config: dict[str, Any], dependencies: TrainingDependencies,
                    coordinator: Coordinator, identity: dict[str, Any], args: Any,
                    model: Any, processor: Any, targets: list[str], admission: Admission,
                    training: list[dict], validation: list[dict]) -> PreparedTrainer:
    """Validate every causal prefix once across ranks, then construct Trainer.

    Keep the causal loss and exact image-aware collator together at this boundary:
    Trainer's token-count contract must match the labels produced by the collator.
    Validation failures are gathered before any rank enters optimization.
    """
    from .collator import DecisionCollator
    from .loss import causal_token_loss

    topology = coordinator.topology
    visual_ids = {getattr(model.config, key, None) for key in
                  ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")} - {None}
    collator = DecisionCollator(processor, max_length=config["max_length"], image_token_ids=visual_ids,
                                token_validation_cache_size=config["token_validation_cache_size"],
                                model_type=getattr(model.config,'model_type',None))
    local_samples = admission.samples[topology.rank::topology.world_size]

    def validate_prefixes() -> int:
        count = 0
        for sample in local_samples:
            encoded = collator([sample])
            count += int((encoded["labels"] != -100).sum().item())
        return count

    token_count = coordinator.sum(coordinator.phase("validate sharded exact multimodal prefixes", validate_prefixes))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    coordinator.phase("validate trainable adapter", lambda: check_trainable(trainable))
    coordinator.agree("LoRA target architecture", {"targets": targets, "trainable_parameters": trainable})
    VerifiedTrainer, FinalCheckpointCallback = trainer_classes(
        dependencies.trainer, dependencies.trainer_callback, coordinator, identity, config,
    )
    trainer = coordinator.phase("construct Trainer", lambda: VerifiedTrainer(
        model=model, args=args, train_dataset=training, eval_dataset=validation or None,
        data_collator=collator, processing_class=processor, compute_loss_func=causal_token_loss,
        callbacks=[FinalCheckpointCallback()],
    ))
    coordinator.phase("validate actual Trainer precision/backend", lambda: check_trainer_runtime(trainer, config))
    shard_counts = coordinator.gather(len(local_samples))
    return PreparedTrainer(trainer, trainable, token_count, shard_counts)
