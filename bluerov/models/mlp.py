"""
Multi-Layer Perceptron for BlueROV2 inverse-model learning.

Input  (8,): [e_surge, e_sway, e_heave, e_psi_sin, e_psi_cos, u, v, w, r]
             body-frame errors  +  current body-frame velocities
Output (4,): [X, Y, Z, M_z]  — required forces/torques
"""

import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Configurable fully-connected network with ReLU hidden layers."""

    def __init__(self, in_dim: int = 9, out_dim: int = 4,
                 hidden: tuple = (128, 128)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
