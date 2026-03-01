"""
Learned Inverse-Model (DNN) Controller for BlueROV2.

A pre-trained MLP maps the current state error + velocities to
the required body-frame forces [X, Y, Z, M_z].

Input features (9,):
    [e_surge, e_sway, e_heave, sin(e_psi), cos(e_psi), u, v, w, r]

Output (4,):
    [X, Y, Z, M_z]  (normalised to [-1, 1], then scaled to physical limits)
"""

import numpy as np
import torch
from ..src.bluerov_model import ssa
from ..src.parameters import MAX_FORCE, MAX_TORQUE
from ..models.mlp import MLP


class DNNBlueROV:
    """
    Feed-forward DNN controller backed by a pre-trained MLP.

    The model weights are fixed (no online learning). Use LoRAController
    for online adaptation variants.
    """

    def __init__(self, model: MLP, dt: float = 0.05,
                 device: str = 'cpu'):
        self.dt     = dt
        self.device = device
        self.model  = model.to(device)
        self.model.eval()

        self.prev_pos_err = np.zeros(4)

    def _build_features(self, state: np.ndarray,
                         ref: np.ndarray) -> torch.Tensor:
        x, y, z = state[0], state[1], state[2]
        cos_psi, sin_psi = state[3], state[4]
        u, v, w, r = state[5], state[6], state[7], state[8]
        psi = np.arctan2(sin_psi, cos_psi)

        e_world = np.array([ref[0] - x, ref[1] - y, ref[2] - z])
        e_surge =  cos_psi * e_world[0] + sin_psi * e_world[1]
        e_sway  = -sin_psi * e_world[0] + cos_psi * e_world[1]
        e_heave =  e_world[2]
        e_psi   = ssa(ref[3] - psi)

        features = np.array([
            e_surge, e_sway, e_heave,
            np.sin(e_psi), np.cos(e_psi),
            u, v, w, r,
        ], dtype=np.float32)
        return torch.from_numpy(features).unsqueeze(0).to(self.device)

    def control(self, state: np.ndarray, ref: np.ndarray) -> np.ndarray:
        feat = self._build_features(state, ref)
        with torch.no_grad():
            out = self.model(feat).squeeze(0).cpu().numpy()
        # Scale from [-1,1] to physical limits
        cmd = np.array([
            out[0] * MAX_FORCE,
            out[1] * MAX_FORCE,
            out[2] * MAX_FORCE,
            out[3] * MAX_TORQUE,
        ])
        cmd[:3] = np.clip(cmd[:3], -MAX_FORCE,  MAX_FORCE)
        cmd[3]  = np.clip(cmd[3],  -MAX_TORQUE, MAX_TORQUE)
        return cmd

    def reset(self):
        self.prev_pos_err[:] = 0.0
