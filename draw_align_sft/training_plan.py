"""Resolve data-epoch budgets after effective batch sizes and topology are known."""
from __future__ import annotations


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_training_plan(config, records, world_size):
    """FM pads each data epoch to complete optimizer updates on every rank.

    Explicit ``max_steps`` is a total optimizer-step budget (including resumed
    steps), and takes precedence over ``num_epochs`` for short validation runs.
    This function does not mutate the requested configuration.
    """
    records = positive_integer(records, "training records")
    world_size = positive_integer(world_size, "world_size")
    options = config["training"]
    batch_size = positive_integer(options.get("batch_size", 1), "training.batch_size")
    accumulation = positive_integer(options["gradient_accumulation"], "training.gradient_accumulation")
    global_batch = world_size * batch_size * accumulation
    steps_per_epoch = (records + global_batch - 1) // global_batch
    explicit_steps = options.get("max_steps")
    epochs = options.get("num_epochs", 5)
    if explicit_steps is None:
        epochs = positive_integer(epochs, "training.num_epochs")
        max_steps = epochs * steps_per_epoch
    else:
        max_steps = positive_integer(explicit_steps, "training.max_steps")
    return {"mode": "epochs" if explicit_steps is None else "steps", "records": records,
            "world_size": world_size, "batch_size": batch_size, "gradient_accumulation": accumulation,
            "global_batch_size": global_batch, "steps_per_epoch": steps_per_epoch,
            "num_epochs": epochs, "max_steps": max_steps,
            "padding_records_per_epoch": steps_per_epoch * global_batch - records}


def resolve_opsd_training_plan(options, sample, records, world_size):
    """Translate data sweeps to upstream outer rounds, preserving K-repeat.

    Upstream's ``num_epochs`` counts rollout/update rounds, not data epochs.
    The epoch sampler pads every final *rollout batch* to complete K-groups;
    the final outer round can include a partial next data epoch.
    """
    records = positive_integer(records, "training records")
    world_size = positive_integer(world_size, "world_size")
    sample_batch = positive_integer(sample["train_batch_size"], "sample.train_batch_size")
    batches = positive_integer(sample["num_batches_per_epoch"], "sample.num_batches_per_epoch")
    k = positive_integer(sample["num_image_per_prompt"], "sample.num_image_per_prompt")
    if k < 2 or sample_batch * world_size % k:
        raise ValueError("sample.train_batch_size * world_size must be divisible by K=num_image_per_prompt >=2")
    prompts_per_batch = sample_batch * world_size // k
    if records < prompts_per_batch:
        raise ValueError(f"Need >= {prompts_per_batch} training rows for upstream K-repeat sampling; found {records}")
    batches_per_data_epoch = (records + prompts_per_batch - 1) // prompts_per_batch
    explicit_rounds = options.get("num_epochs")
    data_epochs = options.get("data_epochs", 5)
    if explicit_rounds is None:
        data_epochs = positive_integer(data_epochs, "upstream.data_epochs")
        rounds = (data_epochs * batches_per_data_epoch + batches - 1) // batches
    else:
        rounds = positive_integer(explicit_rounds, "upstream.num_epochs")
    total_batches = rounds * batches
    return {"mode": "data_epochs" if explicit_rounds is None else "outer_epochs",
            "records": records, "world_size": world_size, "data_epochs": data_epochs,
            "outer_epochs": rounds, "distinct_prompts_per_rollout_batch": prompts_per_batch,
            "rollout_batches_per_data_epoch": batches_per_data_epoch,
            "rollout_batches_per_outer_epoch": batches,
            "completed_data_epochs": total_batches // batches_per_data_epoch,
            "extra_rollout_batches": (total_batches - data_epochs * batches_per_data_epoch)
            if explicit_rounds is None else None,
            "padding_prompt_groups_per_data_epoch": batches_per_data_epoch * prompts_per_batch - records}
