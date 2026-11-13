"""Optimizer adaptation for the upstream trainer's independent student bank."""
from __future__ import annotations


_MUON_DEFAULTS = {
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_lr_adjustment": "match_rms_adamw",
    "muon_aux_adamw_lr": None,
}


def optimizer_options(train):
    """Translate upstream fields without mutating historical run provenance."""
    return {
        "name": train.get("optimizer", "adamw"),
        "lr": train["learning_rate"],
        "betas": (train["adam_beta1"], train["adam_beta2"]),
        "weight_decay": train["adam_weight_decay"],
        "eps": train["adam_epsilon"],
        "momentum": train.get("muon_momentum", _MUON_DEFAULTS["muon_momentum"]),
        "nesterov": train.get("muon_nesterov", _MUON_DEFAULTS["muon_nesterov"]),
        "ns_steps": train.get("muon_ns_steps", _MUON_DEFAULTS["muon_ns_steps"]),
        "muon_lr_adjustment": train.get("muon_lr_adjustment", _MUON_DEFAULTS["muon_lr_adjustment"]),
        "aux_adamw_lr": train.get("muon_aux_adamw_lr"),
    }


def canonical_train_identity(train):
    """Only equate omitted optimizer settings with explicit AdamW defaults.

    Older checkpoints have no optimizer selector. The unchanged default remains
    resumable; selecting Muon or changing any other setting still requires a
    new run. Preserve nondefault Muon fields even when AdamW does not use them.
    """
    identity = dict(train)
    if identity.get("optimizer", "adamw") == "adamw":
        identity.pop("optimizer", None)
        for key, default in _MUON_DEFAULTS.items():
            if key in identity and identity[key] == default and type(identity[key]) is type(default):
                identity.pop(key)
    return identity


def build_student_optimizer(model, student_parameters, train, *, finetune_mode, distributed_mode="ddp"):
    """Optimize precisely the selected student parameters, retaining their order.

    The facade also registers frozen behavior/reference banks. Selecting by
    object identity keeps those banks out of every optimizer parameter group.
    """
    from ..optimizers import build_optimizer

    parameters = list(student_parameters)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("The OPSD student parameter list contains duplicate parameters")
    if any(id(parameter) not in names for parameter in parameters):
        raise ValueError("The OPSD student parameter list contains an unregistered parameter")
    return build_optimizer(
        [(names[id(parameter)], parameter) for parameter in parameters],
        optimizer_options(train),
        finetune_mode=finetune_mode,
        distributed_mode=distributed_mode,
    )
