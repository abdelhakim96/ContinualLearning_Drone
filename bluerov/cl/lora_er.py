"""
LoRA + Experience Replay (LoRA-ER)

Algorithm:
    For each incoming mini-batch (x_new, y_new):
      1. Forward pass through LoRA model.
      2. Compute MSE loss on new data.
      3. Back-propagate (only LoRA params have gradients).
      4. Retrieve a random mini-batch from the replay buffer.
      5. Forward + MSE loss on replay data.
      6. Back-propagate replay loss (accumulates gradient).
      7. Optimiser step.
      8. Add new data to buffer.
"""

import torch
import torch.nn as nn
from ..models.lora import LoRAMLP
from .buffer import ReplayBuffer


class LoRAER:
    """Experience Replay for LoRA adapter parameters."""

    def __init__(self, model: LoRAMLP,
                 lr: float = 3e-4,
                 buffer_size: int = 500,
                 mem_batch: int = 32,
                 device: str = 'cpu'):
        self.model   = model
        self.device  = device
        self.criterion = nn.MSELoss()
        self.buffer  = ReplayBuffer(capacity=buffer_size,
                                    in_dim=9, out_dim=4, device=device)
        self.mem_batch = mem_batch

        # Optimise only LoRA parameters
        self.opt = torch.optim.Adam(model.lora_parameters(), lr=lr)

    def train(self, x: torch.Tensor, y: torch.Tensor):
        """
        One online training step.

        Parameters
        ----------
        x : (B, 9) input features
        y : (B, 4) target normalised controls
        """
        self.model.train()

        # ── New-data loss ──────────────────────────────────────────────────────
        self.opt.zero_grad()
        pred = self.model(x)
        loss_new = self.criterion(pred, y)
        loss_new.backward()

        # ── Replay loss ────────────────────────────────────────────────────────
        mem_x, mem_y = self.buffer.sample(self.mem_batch)
        if mem_x.size(0) > 0:
            pred_mem  = self.model(mem_x)
            loss_mem  = self.criterion(pred_mem, mem_y)
            loss_mem.backward()

        self.opt.step()

        # ── Update buffer ──────────────────────────────────────────────────────
        self.buffer.update(x.detach(), y.detach())

        return loss_new.item()
