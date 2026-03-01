"""
Low-Rank Adaptation (LoRA) for the BlueROV2 inverse model.

LoRA modifies a pre-trained linear layer W ∈ R^{d×k} with
a low-rank correction:

    W_adapted = W  +  (α/r) · B · A

where A ∈ R^{r×k} and B ∈ R^{d×r} with r ≪ min(d, k).
Only A and B are trained; the base weights W are frozen.

References
----------
Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
ICLR 2022.
"""

import math
import torch
import torch.nn as nn
from copy import deepcopy

from .mlp import MLP


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a trainable LoRA correction.

        out = base(x) + scale * B(A(x))
    """

    def __init__(self, base_linear: nn.Linear,
                 rank: int = 4, alpha: float = 1.0):
        super().__init__()
        d = base_linear.out_features
        k = base_linear.in_features

        # Freeze base
        self.base = deepcopy(base_linear)
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank  = rank
        self.scale = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, k))
        self.lora_B = nn.Parameter(torch.zeros(d, rank))

        # Initialise A ~ N(0, σ²), B = 0 → initial Δ = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """Return a plain nn.Linear with LoRA merged into base weights."""
        merged = deepcopy(self.base)
        with torch.no_grad():
            merged.weight.data += self.scale * (self.lora_B @ self.lora_A)
        return merged


class LoRAMLP(nn.Module):
    """
    Wraps a pre-trained MLP and replaces every Linear layer with a LoRALinear.

    The base model weights are frozen; only LoRA parameters are trainable.
    """

    def __init__(self, base_mlp: MLP, rank: int = 4, alpha: float = 1.0):
        super().__init__()

        self.rank  = rank
        self.alpha = alpha

        # Build a mirror of base_mlp.net, replacing Linear → LoRALinear
        adapted_layers = []
        for layer in base_mlp.net:
            if isinstance(layer, nn.Linear):
                adapted_layers.append(LoRALinear(layer, rank=rank, alpha=alpha))
            else:
                adapted_layers.append(deepcopy(layer))
        self.net = nn.Sequential(*adapted_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def lora_parameters(self):
        """Return only the trainable LoRA parameters."""
        for m in self.net.modules():
            if isinstance(m, LoRALinear):
                yield m.lora_A
                yield m.lora_B

    def lora_named_parameters(self):
        """Return (name, param) for all LoRA parameters."""
        for name, m in self.net.named_modules():
            if isinstance(m, LoRALinear):
                yield f"{name}.lora_A", m.lora_A
                yield f"{name}.lora_B", m.lora_B

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.lora_parameters())

    def reset_lora(self):
        """Re-initialise all LoRA adapters to zero correction."""
        for m in self.net.modules():
            if isinstance(m, LoRALinear):
                nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
                nn.init.zeros_(m.lora_B)
