"""
LoRA Continual-Learning Controller for BlueROV2.

Wraps a LoRAMLP model and an associated CL algorithm.
During simulation the controller:
  1. Queries the LoRA model for a control action.
  2. Observes the resulting transition.
  3. Computes a supervision signal from the analytical inverse model
     (used as a "teacher" for the online adaptation target).
  4. Calls the CL algorithm's `train` step.
"""

import numpy as np
import torch
from ..src.bluerov_model import ssa
from ..src.parameters import MAX_FORCE, MAX_TORQUE
from ..models.lora import LoRAMLP
from ..controllers.inverse_bluerov import InverseBlueROV


class LoRACLBlueROV:
    """
    Online continual-learning controller with LoRA adapters.

    Parameters
    ----------
    lora_model  : LoRAMLP instance (base frozen, adapters trainable)
    cl_algo     : an object with a `.train(x, y)` method (e.g. LoRAER)
    dt          : simulation timestep
    device      : 'cpu' or 'cuda'
    """

    def __init__(self, lora_model: LoRAMLP, cl_algo,
                 dt: float = 0.05, device: str = 'cpu'):
        self.dt        = dt
        self.device    = device
        self.model     = lora_model.to(device)
        self.cl_algo   = cl_algo
        self.teacher   = InverseBlueROV(dt=dt)   # analytical teacher signal

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
        """Query LoRA model for control command."""
        feat = self._build_features(state, ref)
        self.model.eval()
        with torch.no_grad():
            out = self.model(feat).squeeze(0).cpu().numpy()
        cmd = np.array([
            out[0] * MAX_FORCE,
            out[1] * MAX_FORCE,
            out[2] * MAX_FORCE,
            out[3] * MAX_TORQUE,
        ])
        cmd[:3] = np.clip(cmd[:3], -MAX_FORCE,  MAX_FORCE)
        cmd[3]  = np.clip(cmd[3],  -MAX_TORQUE, MAX_TORQUE)
        return cmd

    def learn(self, state: np.ndarray, ref: np.ndarray):
        """
        Compute teacher label and trigger one CL training step.

        The teacher is the analytical inverse controller, which provides
        the "correct" control for the current (state, ref) pair.
        """
        feat = self._build_features(state, ref)

        # Teacher label from analytical inverse (normalised to [-1,1])
        teacher_cmd = self.teacher.control(state, ref)
        label = np.array([
            teacher_cmd[0] / MAX_FORCE,
            teacher_cmd[1] / MAX_FORCE,
            teacher_cmd[2] / MAX_FORCE,
            teacher_cmd[3] / MAX_TORQUE,
        ], dtype=np.float32)
        label_t = torch.from_numpy(label).unsqueeze(0).to(self.device)

        self.cl_algo.train(feat, label_t)

    def reset(self):
        self.teacher.reset()
