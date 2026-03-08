"""
AdEMAMix — the most token-efficient optimizer for LLM pretraining.

From Apple (Sep 2024): A 1.3B model trained with AdEMAMix on 101B tokens
matches AdamW trained on 197B tokens — 95% more token-efficient.

Key idea: Adam uses one exponential moving average of gradients. AdEMAMix
uses TWO — a fast EMA (recent gradients, like Adam) and a slow EMA (retains
gradient information from tens of thousands of steps back). This lets the
model "remember" useful gradient signals from much earlier in training.

Reference: https://arxiv.org/abs/2409.03137
"""

import math
import torch
from torch.optim import Optimizer
from typing import Tuple


class AdEMAMix(Optimizer):
    """AdEMAMix: Adam with a mixture of two exponential moving averages.

    Args:
        params: Model parameters
        lr: Learning rate (default: 3e-4)
        betas: Tuple of (beta1_fast, beta2, beta3_slow)
            - beta1_fast: Fast EMA decay for recent gradients (default: 0.9)
            - beta2: Second moment decay, same as Adam (default: 0.999)
            - beta3_slow: Slow EMA decay — the key innovation (default: 0.9999)
        alpha: Mixing coefficient for slow EMA (default: 5.0)
            Higher = more weight on slow (historical) gradients
        T_alpha_beta3: Warmup steps for alpha and beta3 (default: None)
            Ramps alpha from 0 to `alpha` and beta3 from beta1 to `beta3`
            over this many steps. Prevents instability early in training.
        eps: Numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0.1)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        T_alpha_beta3: int = None,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        defaults = dict(
            lr=lr, betas=betas, alpha=alpha,
            T_alpha_beta3=T_alpha_beta3, eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            T_ab = group["T_alpha_beta3"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)  # Fast EMA
                    state["m2"] = torch.zeros_like(p)  # Second moment (like Adam)
                    state["m3"] = torch.zeros_like(p)  # Slow EMA (the innovation)

                m1, m2, m3 = state["m1"], state["m2"], state["m3"]
                state["step"] += 1
                step = state["step"]

                # Schedule alpha and beta3 warmup
                if T_ab is not None and T_ab > 0 and step < T_ab:
                    progress = step / T_ab
                    alpha_t = alpha * progress
                    beta3_t = beta1 + (beta3 - beta1) * progress
                else:
                    alpha_t = alpha
                    beta3_t = beta3

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Update fast EMA (standard Adam momentum)
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Update second moment (standard Adam)
                m2.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Update slow EMA (AdEMAMix innovation)
                m3.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)

                # Bias correction
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                bc3 = 1.0 - beta3_t ** step

                m1_hat = m1 / bc1
                m2_hat = m2 / bc2
                m3_hat = m3 / bc3

                # Mix fast and slow EMAs
                # This is the key: the update direction combines recent
                # gradient info (m1) with long-term gradient memory (m3)
                mixed = m1_hat + alpha_t * m3_hat

                # Adam-style update with mixed momentum
                denom = m2_hat.sqrt().add_(group["eps"])
                p.addcdiv_(mixed, denom, value=-group["lr"])

        return loss


class Lion(Optimizer):
    """Lion optimizer — discovered by Google Brain via AutoML.

    Uses only the sign of the gradient, halving optimizer memory vs Adam.
    Requires 3-10x smaller learning rate than AdamW.

    Reference: https://arxiv.org/abs/2302.06675
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Update: use sign of interpolation between gradient and momentum
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign_(), alpha=-group["lr"])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
