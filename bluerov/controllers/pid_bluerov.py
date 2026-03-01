"""
Hierarchical PID Controller for BlueROV2 4-DOF trajectory tracking.

Outer loop: position error  → desired body-frame velocities
Inner loop: velocity error  → body-frame forces / torques [X, Y, Z, M_z]
"""

import numpy as np
from ..src.bluerov_model import ssa
from ..src.parameters import MAX_FORCE, MAX_TORQUE


class PIDBlueROV:
    """
    Two-level PID controller.

    Level 1 — Position controller
        Inputs : position error in world frame [e_x, e_y, e_z] + heading error
        Outputs: desired body-frame velocities  [u_d, v_d, w_d, r_d]

    Level 2 — Velocity controller
        Inputs : velocity error [e_u, e_v, e_w, e_r]
        Outputs: forces / torques [X, Y, Z, M_z]
    """

    def __init__(self, dt: float = 0.05):
        self.dt = dt

        # ── Level-1 gains (position → velocity setpoint) ──────────────────────
        self.Kp_pos  = np.array([1.5,  1.5,  1.2,  2.0])   # P gains [x,y,z,ψ]
        self.Ki_pos  = np.array([0.05, 0.05, 0.03, 0.08])   # I gains
        self.Kd_pos  = np.array([0.3,  0.3,  0.2,  0.4])    # D gains

        # ── Level-2 gains (velocity → force / torque) ─────────────────────────
        self.Kp_vel  = np.array([40.0, 40.0, 30.0, 6.0])
        self.Ki_vel  = np.array([2.0,  2.0,  1.5,  0.3])
        self.Kd_vel  = np.array([8.0,  8.0,  6.0,  1.2])

        # ── Integrator states ─────────────────────────────────────────────────
        self.int_pos = np.zeros(4)    # ∫ position errors
        self.int_vel = np.zeros(4)    # ∫ velocity errors
        self.prev_pos_err = np.zeros(4)
        self.prev_vel     = np.zeros(4)

        # ── Anti-windup bounds ────────────────────────────────────────────────
        self.int_pos_lim = np.array([2.0, 2.0, 1.5, 1.0])
        self.int_vel_lim = np.array([5.0, 5.0, 4.0, 1.0])

    def _world_to_body(self, e_world: np.ndarray,
                       cos_psi: float, sin_psi: float) -> np.ndarray:
        """Rotate [e_x, e_y, e_z] from world to body frame."""
        e_surge =  cos_psi * e_world[0] + sin_psi * e_world[1]
        e_sway  = -sin_psi * e_world[0] + cos_psi * e_world[1]
        e_heave =  e_world[2]
        return np.array([e_surge, e_sway, e_heave])

    def control(self, state: np.ndarray,
                ref: np.ndarray) -> np.ndarray:
        """
        Compute control action.

        Parameters
        ----------
        state : (9,) [x,y,z,cos_ψ,sin_ψ,u,v,w,r]
        ref   : (4,) [x_ref, y_ref, z_ref, ψ_ref]

        Returns
        -------
        cmd : (4,) [X, Y, Z, M_z]
        """
        x, y, z = state[0], state[1], state[2]
        cos_psi, sin_psi = state[3], state[4]
        u, v, w, r = state[5], state[6], state[7], state[8]

        psi = np.arctan2(sin_psi, cos_psi)

        # ── Level-1: position errors ───────────────────────────────────────────
        e_world = np.array([ref[0] - x, ref[1] - y, ref[2] - z])
        e_body3 = self._world_to_body(e_world, cos_psi, sin_psi)
        e_psi   = ssa(ref[3] - psi)
        pos_err = np.array([e_body3[0], e_body3[1], e_body3[2], e_psi])

        # Derivative of position error
        d_pos_err = (pos_err - self.prev_pos_err) / self.dt
        self.prev_pos_err = pos_err.copy()

        # Integrator with anti-windup
        self.int_pos = np.clip(self.int_pos + pos_err * self.dt,
                               -self.int_pos_lim, self.int_pos_lim)

        # Desired velocities (body frame)
        vel_ref = (self.Kp_pos * pos_err
                   + self.Ki_pos * self.int_pos
                   + self.Kd_pos * d_pos_err)

        # ── Level-2: velocity errors ───────────────────────────────────────────
        vel_cur = np.array([u, v, w, r])
        vel_err = vel_ref - vel_cur

        d_vel = (vel_cur - self.prev_vel) / self.dt
        self.prev_vel = vel_cur.copy()

        self.int_vel = np.clip(self.int_vel + vel_err * self.dt,
                               -self.int_vel_lim, self.int_vel_lim)

        cmd = (self.Kp_vel * vel_err
               + self.Ki_vel * self.int_vel
               - self.Kd_vel * d_vel)

        # Clip to physical limits
        cmd[:3] = np.clip(cmd[:3], -MAX_FORCE,  MAX_FORCE)
        cmd[3]  = np.clip(cmd[3],  -MAX_TORQUE, MAX_TORQUE)

        return cmd

    def reset(self):
        self.int_pos[:] = 0.0
        self.int_vel[:] = 0.0
        self.prev_pos_err[:] = 0.0
        self.prev_vel[:] = 0.0
