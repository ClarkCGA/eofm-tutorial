
"""
optimizer.py — Rewritten
------------------------
A clean, extensible optimizer & scheduler factory with first-class support for
AdamW (standalone or as SAM's base optimizer), safe argument handling, and
ergonomic YAML-friendly config keys.

Public API
----------
- build_optimizer(model, config) -> (optimizer, scheduler)
  * config: dict-like (e.g., loaded from YAML)
    Keys (common):
      optimizer: one of {"sgd","adam","adamw","rmsprop","adagrad","adadelta","nadam","adamax","lbfgs","sam"}
      learning_rate_init (float): initial LR
      weight_decay (float): weight decay (L2 or decoupled depending on optimizer)
      momentum (float): only applied for optimizers that accept it
      nesterov (bool): only for SGD variants that support it
      betas: [b1, b2] for Adam/AdamW/NAdam/Adamax (optional)

    SAM-specific:
      base_optimizer: same choices as above (default: "sgd")
      rho (float): SAM neighborhood size (default: 0.05)
      adaptive (bool): ASAM variant (default: False)

    Scheduler:
      scheduler: one of {"none","cosine","cosinewarm","step","multistep","plateau","poly","onecycle"}
      # cosine:
      T_max, eta_min
      # cosinewarm (CosineAnnealingWarmRestarts):
      t_0, t_mult, eta_min
      # step:
      step_size, gamma
      # multistep:
      milestones (list[int]), gamma
      # plateau:
      mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose
      # poly (PolynomialLR below):
      max_decay_steps, min_learning_rate, power
      # onecycle:
      max_lr, total_steps OR (epochs and steps_per_epoch), pct_start, anneal_strategy, div_factor, final_div_factor

Notes
-----
- Momentum is *only* passed to optimizers whose signature supports it.
- If you use SAM, the scheduler is attached to the *base* optimizer, which is the
  standard practice in SAM training loops.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, List
import inspect

import torch
from torch.optim import Optimizer
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

# -----------------------------
# PolynomialLR (kept from prior)
# -----------------------------
class PolynomialLR(lrs._LRScheduler):
    """
    Polynomial learning rate decay until step reaches max_decay_steps.

    Args:
        optimizer: Wrapped optimizer.
        max_decay_steps (int): after this step, stop decreasing learning rate.
        min_learning_rate (float): floor LR.
        power (float): power of the polynomial.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        max_decay_steps: int,
        min_learning_rate: float = 1e-6,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        if max_decay_steps <= 0:
            raise ValueError("max_decay_steps must be > 0")
        self.max_decay_steps = int(max_decay_steps)
        self.min_learning_rate = float(min_learning_rate)
        self.power = float(power)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = min(self.last_epoch, self.max_decay_steps)
        if self.max_decay_steps == 0:
            factor = 0.0
        else:
            factor = (1.0 - step / self.max_decay_steps) ** self.power

        lrs_out = []
        for base_lr in self.base_lrs:
            new_lr = (base_lr - self.min_learning_rate) * factor + self.min_learning_rate
            lrs_out.append(new_lr)
        return lrs_out


# SAM optimizer

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) wrapper.

    Usage:
        base_opt = optim.SGD | optim.AdamW | ...
        optimizer = SAM(model.parameters(),
                        base_optimizer=optim.AdamW,
                        rho=0.05,
                        adaptive=False,
                        lr=3e-4, weight_decay=0.05)

    You must call .first_step(...) then .second_step(...), or use the provided
    .step(closure) pattern below (closure should compute loss and call backward).
    """
    def __init__(
        self,
        params: Iterable,
        base_optimizer: Any,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        if isinstance(base_optimizer, type):
            base_opt_cls = base_optimizer
        else:
            # If a function/callable was passed
            base_opt_cls = base_optimizer

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_opt_cls(self.param_groups, **kwargs)
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norm_sq = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = p.abs() if self.adaptive else 1.0
                norm_sq.add_(torch.sum((scale * p.grad) ** 2))
        return torch.sqrt(norm_sq + 1e-12)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (p.abs() if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                # Save perturbation for second step
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p].pop("e_w", torch.zeros_like(p)))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """
        SAM requires a closure:
            def closure():
                loss = criterion(model(input), target)
                loss.backward()
                return loss
        """
        assert closure is not None, "SAM.step() requires a closure that computes the loss and calls backward()"
        # First forward-backward pass
        loss = closure()
        self.first_step(zero_grad=True)
        # Second forward-backward pass at perturbed weights
        closure()
        self.second_step(zero_grad=True)
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    # Expose param_groups etc. to behave like an optimizer
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
    @param_groups.setter
    def param_groups(self, value):
        # When Optimizer.__init__ calls add_param_group, it sets this attribute.
        # We forward to base optimizer if it already exists.
        if hasattr(self, "base_optimizer"):
            self.base_optimizer.param_groups = value
        else:
            super(SAM, self.__class__).param_groups.fset(self, value)  # type: ignore


# Optimizer lookup helper

def _lookup_optim(name: str):
    name = name.lower()
    aliases = {
        "sgd": "SGD",
        "adam": "Adam",
        "adamw": "AdamW",
        "rmsprop": "RMSprop",
        "adagrad": "Adagrad",
        "adadelta": "Adadelta",
        "nadam": "NAdam",
        "adamax": "Adamax",
        "lbfgs": "LBFGS",
    }
    cls_name = aliases.get(name, name)
    if not hasattr(optim, cls_name):
        raise ValueError(f"Unknown optimizer '{name}' (resolved to '{cls_name}')")
    return getattr(optim, cls_name)

def _maybe_add(param_dict: Dict[str, Any], key: str, value: Any, opt_class) -> None:
    """
    Insert (key,value) into param_dict if the optimizer's __init__ supports it.
    """
    sig = inspect.signature(opt_class.__init__)
    if key in sig.parameters:
        param_dict[key] = value


# Scheduler factory

def _build_scheduler(config: Dict[str, Any], optimizer: Optimizer) -> Optional[lrs._LRScheduler]:
    sched_name = str(config.get("scheduler", "none")).lower()
    if sched_name in ("none", "off", "disable", "disabled", "no"):
        return None

    if sched_name == "cosine":
        T_max = int(config.get("T_max", 100))
        eta_min = float(config.get("eta_min", 0.0))
        return lrs.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    if sched_name == "cosinewarm":
        t_0 = int(config.get("t_0", 10))
        t_mult = int(config.get("t_mult", 1))
        eta_min = float(config.get("eta_min", 0.0))
        return lrs.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)

    if sched_name == "step":
        step_size = int(config.get("step_size", 30))
        gamma = float(config.get("gamma", 0.1))
        return lrs.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if sched_name == "multistep":
        milestones = config.get("milestones", [30, 60, 90])
        gamma = float(config.get("gamma", 0.1))
        return lrs.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if sched_name == "plateau":
        # Common defaults
        return lrs.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "min")),
            factor=float(config.get("factor", 0.1)),
            patience=int(config.get("patience", 10)),
            threshold=float(config.get("threshold", 1e-4)),
            threshold_mode=str(config.get("threshold_mode", "rel")),
            cooldown=int(config.get("cooldown", 0)),
            min_lr=float(config.get("min_lr", 0.0)),
            eps=float(config.get("eps", 1e-8)),
            verbose=bool(config.get("verbose", False)),
        )

    if sched_name == "poly":
        return PolynomialLR(
            optimizer,
            max_decay_steps=int(config.get("max_decay_steps", 100)),
            min_learning_rate=float(config.get("min_learning_rate", 1e-6)),
            power=float(config.get("power", 1.0)),
        )

    if sched_name == "onecycle":
        # OneCycle needs either total_steps OR (epochs and steps_per_epoch)
        kwargs = dict(
            max_lr=float(config.get("max_lr", 1e-3)),
            pct_start=float(config.get("pct_start", 0.3)),
            anneal_strategy=str(config.get("anneal_strategy", "cos")),
            div_factor=float(config.get("div_factor", 25.0)),
            final_div_factor=float(config.get("final_div_factor", 1e4)),
        )
        if "total_steps" in config:
            kwargs["total_steps"] = int(config["total_steps"])
        else:
            kwargs["epochs"] = int(config.get("epochs", 100))
            kwargs["steps_per_epoch"] = int(config.get("steps_per_epoch", 100))
        return lrs.OneCycleLR(optimizer, **kwargs)

    raise ValueError(f"Unknown scheduler: {sched_name}")


# Public factory

def get_optimizer(model, config: Dict[str, Any]) -> Tuple[Optimizer, Optional[lrs._LRScheduler]]:
    """
    Create optimizer and scheduler from a config dict.

    Returns:
        (optimizer, scheduler)
    """
    name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate_init", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    momentum = float(config.get("momentum", 0.9))
    nesterov = bool(config.get("nesterov", False))
    betas = config.get("betas", None)

    if name == "sam":
        base_name = str(config.get("base_optimizer", "sgd")).lower()
        base_cls = _lookup_optim(base_name)

        base_kwargs = {"lr": lr, "weight_decay": weight_decay}
        # Conditionally add momentum/nesterov/betas if supported
        _maybe_add(base_kwargs, "momentum", momentum, base_cls)
        _maybe_add(base_kwargs, "nesterov", nesterov, base_cls)
        if betas is not None:
            _maybe_add(base_kwargs, "betas", tuple(betas), base_cls)

        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_cls,
            rho=float(config.get("rho", 0.05)),
            adaptive=bool(config.get("adaptive", False)),
            **base_kwargs,
        )
        # schedulers should attach to the base optimizer's param groups
        sched_target = optimizer.base_optimizer

    else:
        opt_cls = _lookup_optim(name)
        opt_kwargs = {"lr": lr, "weight_decay": weight_decay}
        _maybe_add(opt_kwargs, "momentum", momentum, opt_cls)
        _maybe_add(opt_kwargs, "nesterov", nesterov, opt_cls)
        if betas is not None:
            _maybe_add(opt_kwargs, "betas", tuple(betas), opt_cls)
        optimizer = opt_cls(model.parameters(), **opt_kwargs)
        sched_target = optimizer

    scheduler = _build_scheduler(config, sched_target)
    return optimizer, scheduler
