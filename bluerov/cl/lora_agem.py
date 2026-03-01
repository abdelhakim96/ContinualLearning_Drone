"""
LoRA + Averaged Gradient Episodic Memory (LoRA-A-GEM)

A-GEM projects the current gradient g onto the half-space defined by
the reference gradient g_ref (computed on episodic memory), ensuring
that loss on past tasks does not increase.

    if g · g_ref < 0:
        g_projected = g − (g · g_ref / ||g_ref||²) g_ref
    else:
        g_projected = g          (no interference detected)

References
----------
Chaudhry, A. et al. (2019). Efficient Lifelong Learning with A-GEM.
ICLR 2019.
"""

import torch
import torch.nn as nn
from ..models.lora import LoRAMLP
from .buffer import ReplayBuffer


class LoRAAGEM:
    """A-GEM continual learning for LoRA adapter parameters."""

    def __init__(self, model: LoRAMLP,
                 lr: float = 3e-4,
                 buffer_size: int = 500,
                 mem_batch: int = 32,
                 device: str = 'cpu'):
        self.model     = model
        self.device    = device
        self.criterion = nn.MSELoss()
        self.buffer    = ReplayBuffer(capacity=buffer_size,
                                      in_dim=9, out_dim=4, device=device)
        self.mem_batch = mem_batch
        self.opt       = torch.optim.Adam(model.lora_parameters(), lr=lr)

    def _get_lora_grad_vector(self) -> torch.Tensor:
        grads = []
        for p in self.model.lora_parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=self.device))
        return torch.cat(grads)

    def _set_lora_grad_vector(self, g: torch.Tensor):
        offset = 0
        for p in self.model.lora_parameters():
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(g[offset:offset+n].view(p.shape))
            offset += n

    def train(self, x: torch.Tensor, y: torch.Tensor):
        self.model.train()

        # ── Compute gradient on new data ───────────────────────────────────────
        self.opt.zero_grad()
        pred     = self.model(x)
        loss_new = self.criterion(pred, y)
        loss_new.backward()
        g_new = self._get_lora_grad_vector().clone()

        # ── Compute reference gradient on episodic memory ─────────────────────
        mem_x, mem_y = self.buffer.sample(self.mem_batch)
        if mem_x.size(0) > 0:
            self.opt.zero_grad()
            pred_mem = self.model(mem_x)
            loss_mem = self.criterion(pred_mem, mem_y)
            loss_mem.backward()
            g_ref = self._get_lora_grad_vector().clone()

            # ── Project gradient if violating ─────────────────────────────────
            dot = torch.dot(g_new, g_ref)
            if dot < 0:
                g_proj = g_new - (dot / (torch.dot(g_ref, g_ref) + 1e-8)) * g_ref
                self.opt.zero_grad()
                self._set_lora_grad_vector(g_proj)
            else:
                self.opt.zero_grad()
                self._set_lora_grad_vector(g_new)
        else:
            self._set_lora_grad_vector(g_new)

        self.opt.step()
        self.buffer.update(x.detach(), y.detach())
        return loss_new.item()
