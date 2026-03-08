"""Analytical inverse-dynamics controller for the BlueROV2.

Computes the forces required to achieve a desired acceleration
that drives the state toward the reference trajectory.
"""
import numpy as np
from survey.src.parameters import (m, X_ud, Y_vd, Z_wd, N_rd, I_zz,
                                     X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc,
                                     N_r, N_rc, g, F_bouy,
                                     MAX_FORCE, MAX_TORQUE)
from survey.src.bluerov_sim import ssa


class InverseController:
    """PD + analytical inverse dynamics."""

    def __init__(self, dt=0.05, Kp_pos=2.0, Kd_pos=4.0,
                 Kp_psi=3.0, Kd_psi=1.5):
        self.dt = dt
        self.Kp_pos = Kp_pos; self.Kd_pos = Kd_pos
        self.Kp_psi = Kp_psi; self.Kd_psi = Kd_psi

    def control(self, x, ref):
        """
        x   : state (8,) = [x,y,z,psi,u,v,w,r]
        ref : (4,) = [x_d, y_d, z_d, psi_d]
        """
        psi = x[3]
        u_b, v_b, w_b, r_b = x[4], x[5], x[6], x[7]
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # Position error in body frame
        e_n = ref[0] - x[0]
        e_e = ref[1] - x[1]
        e_d = ref[2] - x[2]
        e_surge =  cpsi * e_n + spsi * e_e
        e_sway  = -spsi * e_n + cpsi * e_e
        e_heave = e_d
        e_psi   = ssa(ref[3] - psi)

        # Desired accelerations (PD)
        a_u = self.Kp_pos * e_surge - self.Kd_pos * u_b
        a_v = self.Kp_pos * e_sway  - self.Kd_pos * v_b
        a_w = self.Kp_pos * e_heave - self.Kd_pos * w_b
        a_r = self.Kp_psi * e_psi   - self.Kd_psi * r_b

        # Inverse dynamics
        Mu = m - X_ud
        Mv = m - Y_vd
        Mw = m - Z_wd
        Jr = I_zz - N_rd

        Fx = Mu * a_u - (m - Y_vd) * v_b * r_b - (X_u + X_uc * abs(u_b)) * u_b
        Fy = Mv * a_v + (m - X_ud) * u_b * r_b - (Y_v + Y_vc * abs(v_b)) * v_b
        Fz = Mw * a_w - (Z_w + Z_wc * abs(w_b)) * w_b - m * g + F_bouy
        Mz = Jr * a_r + (X_ud - Y_vd) * u_b * v_b - (N_r + N_rc * abs(r_b)) * r_b

        cmd = np.array([Fx, Fy, Fz, Mz])
        cmd[:3] = np.clip(cmd[:3], -MAX_FORCE, MAX_FORCE)
        cmd[3]  = np.clip(cmd[3], -MAX_TORQUE, MAX_TORQUE)
        return cmd
