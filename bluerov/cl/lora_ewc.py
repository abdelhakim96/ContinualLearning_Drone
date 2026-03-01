"""
LoRA + Elastic Weight Consolidation (LoRA-EWC)

At the end of each operational phase, the Fisher Information Matrix (FIM)
is estimated for the LoRA parameters. Future gradient steps are penalised
proportionally to the FIM, preventing forgetting of previously learned
disturbance compensation.

Loss = L_task(θ_LoRA) + (λ/2) Σ_i F_i (θ_i − θ*_i)²

References
----------
Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural
networks. PNAS, 114(13), 3521–3526.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional

from ..models.lora import LoRAMLP
from .buffer import ReplayBuffer


class LoRAEWC:
    """EWC regularisation applied to LoRA adapter parameters."""

    def __init__(self, model: LoRAMLP,
                 lr: float = 3e-4,
                 ewc_lambda: float = 5000.0,
                 buffer_size: int = 500,
                 mem_batch: int = 32,
                 device: str = 'cpu'):
        self.model      = model
        self.device     = device
        self.lam        = ewc_lambda
        self.criterion  = nn.MSELoss()
        self.buffer     = ReplayBuffer(capacity=buffer_size,
                                       in_dim=9, out_dim=4, device=device)
        self.mem_batch  = mem_batch

        self.opt = torch.optim.Adam(model.lora_parameters(), lr=lr)

        # EWC state: Fisher diagonals + consolidated parameters
        self.fisher: Optional[dict] = None
        self.theta_star: Optional[dict] = None

    # ── EWC consolidation ──────────────────────────────────────────────────────
    def consolidate(self, data_x: torch.Tensor, data_y: torch.Tensor,
                    n_samples: int = 200):
        """
        Estimate diagonal Fisher Information Matrix on a data snapshot
        and store current LoRA parameters as θ*.

        Call this at the end of each operational phase.

        Parameters
        ----------
        data_x   : representative input batch
        data_y   : corresponding targets
        n_samples: number of samples to use for FIM estimation
        """
        self.model.eval()
        fisher = {name: torch.zeros_like(p)
                  for name, p in self.model.lora_named_parameters()}
        theta_star = {name: p.clone().detach()
                      for name, p in self.model.lora_named_parameters()}

        n = min(n_samples, data_x.size(0))
        idx = torch.randperm(data_x.size(0))[:n]
        x_s = data_x[idx]
        y_s = data_y[idx]

        for i in range(n):
            self.opt.zero_grad()
            out  = self.model(x_s[i:i+1])
            loss = self.criterion(out, y_s[i:i+1])
            loss.backward()
            for name, p in self.model.lora_named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.data ** 2

        # Normalise
        for name in fisher:
            fisher[name] /= n

        # Accumulate: new FIM replaces old (or add for task-incremental)
        if self.fisher is None:
            self.fisher    = fisher
            self.theta_star = theta_star
        else:
            for name in fisher:
                self.fisher[name]    += fisher[name]
                self.theta_star[name] = theta_star[name]

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        if self.fisher is None:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for name, p in self.model.lora_named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (p - self.theta_star[name]) ** 2).sum()
        return (self.lam / 2) * loss

    # ── Online training step ───────────────────────────────────────────────────
    def train(self, x: torch.Tensor, y: torch.Tensor):
        self.model.train()

        self.opt.zero_grad()
        pred     = self.model(x)
        loss_new = self.criterion(pred, y)
        ewc_pen  = self._ewc_penalty()
        total    = loss_new + ewc_pen
        total.backward()

        # Replay
        mem_x, mem_y = self.buffer.sample(self.mem_batch)
        if mem_x.size(0) > 0:
            pred_mem = self.model(mem_x)
            loss_mem = self.criterion(pred_mem, mem_y)
            loss_mem.backward()

        self.opt.step()
        self.buffer.update(x.detach(), y.detach())

        return loss_new.item()
