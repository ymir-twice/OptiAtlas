"""AdamW and experimental factor-wise LoRA Muon, on unsharded parameters.

Muon equations/NS kernel follow PyTorch v2.9.0 torch/optim/_muon.py
(BSD, see third_party/muon/LICENSE-PyTorch), derived from Keller Jordan's
Muon (MIT, see third_party/muon/LICENSE-Muon). This compatibility path is
fixed across torch versions; installing a newer torch does not change it.

DDP already reduces gradients, so each rank makes the same local update.
FSDP shards are NOT valid matrices for this algorithm. LoRA A and B are
optimized separately; this is not a low-rank-manifold/Riemannian optimizer.
"""
from __future__ import annotations

import math
import torch


def validate_optimizer_config(options, *, finetune_mode, distributed_mode):
    name = str(options.get("name", "adamw")).lower()
    if name not in {"adamw", "muon"}:
        raise ValueError("optimizer.name must be adamw or muon")
    if name == "adamw":
        return
    if finetune_mode != "lora":
        raise ValueError("Muon currently supports LoRA only; full-parameter/FSDP Muon is not implemented")
    if distributed_mode not in {"auto", "ddp", "none"}:
        raise ValueError("Muon requires unsharded parameters (DDP/none); FSDP Muon is not implemented")
    for key, default in (("lr", 1e-5), ("weight_decay", .01), ("eps", 1e-8)):
        value = float(options.get(key, default))
        if not math.isfinite(value) or value < 0 or (key == "eps" and value == 0):
            raise ValueError(f"optimizer.{key} must be finite and {'positive' if key == 'eps' else 'nonnegative'}")
    aux_lr = options.get("aux_adamw_lr")
    if aux_lr is not None and (not math.isfinite(float(aux_lr)) or float(aux_lr) < 0):
        raise ValueError("optimizer.aux_adamw_lr must be finite and nonnegative")
    momentum = float(options.get("momentum", .95))
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("optimizer.momentum must be in [0, 1)")
    if not isinstance(options.get("nesterov", True), bool):
        raise ValueError("optimizer.nesterov must be a boolean")
    steps = options.get("ns_steps", 5)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps < 100:
        raise ValueError("optimizer.ns_steps must be an integer in [1, 99]")
    if options.get("muon_lr_adjustment", "match_rms_adamw") not in {"original", "match_rms_adamw"}:
        raise ValueError("optimizer.muon_lr_adjustment must be original or match_rms_adamw")
    betas = options.get("betas", (.9, .999))
    if len(betas) != 2 or any(not math.isfinite(float(b)) or not 0 <= float(b) < 1 for b in betas):
        raise ValueError("optimizer.betas must contain two values in [0, 1)")


def _orthogonalize(update, steps):
    # clone protects gradients/momentum if they already have BF16 dtype.
    matrix = update.to(dtype=torch.bfloat16, copy=True)
    transpose = matrix.shape[0] > matrix.shape[1]
    if transpose:
        matrix = matrix.T
    matrix.div_(matrix.norm().clamp(min=1e-7))
    for _ in range(steps):
        gram = matrix @ matrix.T
        polynomial = torch.addmm(gram, gram, gram, beta=-4.7750, alpha=2.0315)
        matrix = torch.addmm(matrix, polynomial, matrix, beta=3.4445)
    return matrix.T if transpose else matrix


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """One scheduler/checkpoint/GradScaler interface for two parameter groups."""

    def __init__(self, groups):
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate all gradients before modifying any parameters.
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is not None and (grad.is_sparse or parameter.is_complex()):
                    raise ValueError("Muon/auxiliary AdamW require dense, real gradients")
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                lr = group["lr"]
                if group["use_muon"]:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    buffer = state["momentum_buffer"]
                    buffer.lerp_(grad, 1 - group["momentum"])
                    update = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                    # Autocast must not silently change the NS kernel arithmetic.
                    with torch.autocast(device_type=parameter.device.type, enabled=False):
                        direction = _orthogonalize(update, group["ns_steps"])
                    rows, columns = parameter.shape
                    scale = (.2 * math.sqrt(max(rows, columns))
                             if group["muon_lr_adjustment"] == "match_rms_adamw"
                             else math.sqrt(max(1., rows / columns)))
                    parameter.mul_(1 - lr * group["weight_decay"])
                    parameter.add_(direction, alpha=-lr * scale)
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    step = state["step"]
                    denominator = state["exp_avg_sq"].sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                    parameter.mul_(1 - lr * group["weight_decay"])
                    parameter.addcdiv_(state["exp_avg"], denominator, value=-lr / (1 - beta1 ** step))
        return loss

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict ordinarily zips by position without checking
        # names/shapes. Never silently import AdamW state or reordered LoRA state.
        saved_groups = state_dict.get("param_groups", [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("Muon checkpoint parameter groups differ")
        for saved, current in zip(saved_groups, self.param_groups):
            for key in ("optimizer_format", "use_muon", "param_names", "param_shapes"):
                if saved.get(key) != current[key]:
                    raise ValueError(f"Muon checkpoint {key} differs; start a new run when switching optimizers/parameters")
            for param_id, parameter in zip(saved["params"], current["params"]):
                state = state_dict.get("state", {}).get(param_id)
                if state is None:
                    continue  # Parameters that have never received a gradient.
                expected = {"step", "momentum_buffer"} if current["use_muon"] else {"step", "exp_avg", "exp_avg_sq"}
                if set(state) != expected or not isinstance(state["step"], int) or state["step"] < 1:
                    raise ValueError("Invalid Muon checkpoint optimizer state")
                for key in expected - {"step"}:
                    if not isinstance(state[key], torch.Tensor) or state[key].shape != parameter.shape:
                        raise ValueError("Muon checkpoint optimizer state shape differs")
        return super().load_state_dict(state_dict)


def build_optimizer(named_parameters, options, *, finetune_mode, distributed_mode):
    validate_optimizer_config(options, finetune_mode=finetune_mode, distributed_mode=distributed_mode)
    named = [(name, param) for name, param in named_parameters if param.requires_grad]
    if not named:
        raise ValueError("Optimizer received no trainable parameters")
    if len({id(p) for _, p in named}) != len(named) or len({n for n, _ in named}) != len(named):
        raise ValueError("Optimizer parameter names and objects must be unique")
    common = dict(lr=float(options["lr"]), weight_decay=float(options["weight_decay"]),
                  betas=tuple(options["betas"]), eps=float(options["eps"]))
    if str(options.get("name", "adamw")).lower() == "adamw":
        # Preserve existing numerics, parameter ordering, and checkpoint format.
        return torch.optim.AdamW([p for _, p in named], **common)
    muon, auxiliary = [], []
    for name, parameter in named:
        if hasattr(parameter, "_local_tensor") or hasattr(parameter, "_fsdp_flattened"):
            raise ValueError("Muon cannot operate on FSDP/DTensor parameter shards")
        is_factor = bool({"lora_A", "lora_B"}.intersection(name.split(".")))
        if is_factor and parameter.ndim != 2:
            raise ValueError(f"Muon LoRA factor must be a 2D matrix: {name}")
        (muon if is_factor else auxiliary).append((name, parameter))
    if not muon:
        raise ValueError("Muon found no trainable 2D LoRA lora_A/lora_B matrices")
    groups = []
    for use_muon, items in ((True, muon), (False, auxiliary)):
        if not items:
            continue
        group = {**common, "params": [p for _, p in items], "use_muon": use_muon,
                 "param_names": [n for n, _ in items], "param_shapes": [list(p.shape) for _, p in items],
                 "optimizer_format": "draw_align_muon_v1"}
        if use_muon:
            group.update(momentum=float(options.get("momentum", .95)),
                         nesterov=options.get("nesterov", True), ns_steps=options.get("ns_steps", 5),
                         muon_lr_adjustment=options.get("muon_lr_adjustment", "match_rms_adamw"))
        elif options.get("aux_adamw_lr") is not None:
            group["lr"] = float(options["aux_adamw_lr"])
        groups.append(group)
    return MuonWithAuxAdamW(groups)


def optimizer_summary(optimizer):
    is_muon = isinstance(optimizer, MuonWithAuxAdamW)
    groups = []
    for group in optimizer.param_groups:
        item = {"kind": "muon" if group.get("use_muon", False) else "adamw",
                "tensors": len(group["params"]), "parameters": sum(p.numel() for p in group["params"]),
                "lr": group["lr"], "weight_decay": group["weight_decay"]}
        if group.get("use_muon", False):
            item.update({key: group[key] for key in ("momentum", "nesterov", "ns_steps", "muon_lr_adjustment")})
        groups.append(item)
    return {"kind": "muon" if is_muon else "adamw", "implementation": type(optimizer).__name__,
            "groups": groups, "scope": "factor_wise_lora" if is_muon else "all_trainable"}
