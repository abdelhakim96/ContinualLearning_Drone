"""
CasADi symbolic dynamics model for the BlueROV2 Heavy 4-DOF AUV.

State:   x = [x_pos, y_pos, z_pos, psi, u_surge, v_sway, w_heave, r_yaw]
Control: u = [X, Y, Z, Mz]   (body-frame thruster forces and yaw torque)
Param:   p = [d0,...,d7]      (additive continuous-time disturbance, optional)

The CasADi absolute value ca.fabs() is used for quadratic drag — this is
C1-differentiable everywhere and directly differentiable by CasADi.

This module also builds convenience CasADi Functions for:
  - f_expl_fn   : evaluates f(x,u,p)
  - f_jac_x_fn  : Jacobian ∂f/∂x  (used for LQR warm-start / debugging)
  - f_jac_u_fn  : Jacobian ∂f/∂u
"""

import os
import numpy as np
import casadi as ca

from .params import VehicleParams, NMPCParams
from .interfaces import DynamicsModel


class BlueROVCasadiModel(DynamicsModel):
    """
    Fully symbolic BlueROV2 dynamics for use in acados NMPC.

    Parameters
    ----------
    vp : VehicleParams
        Physical parameters.  Defaults to calibrated BlueROV2 Heavy values.
    with_disturbance : bool
        If True, a parameter vector p ∈ R^8 is added to the rhs as an
        additive continuous-time disturbance estimate (from a learned model).
        If False, p is empty (pure analytical NMPC).
    """

    nx: int = 8
    nu: int = 4

    def __init__(self,
                 vp: VehicleParams = None,
                 with_disturbance: bool = True):
        self.vp = vp or VehicleParams()
        self.with_disturbance = with_disturbance
        self.np = self.nx if with_disturbance else 0

        # Build symbolic expressions once; cache them
        self._x_sym  = None
        self._u_sym  = None
        self._p_sym  = None
        self._f_expl = None
        self._build()

    # ── Public interface (implements DynamicsModel) ───────────────────────────

    def get_casadi_model(self):
        return self._x_sym, self._u_sym, self._p_sym, self._f_expl

    def get_name(self) -> str:
        suffix = '_dist' if self.with_disturbance else ''
        return f'bluerov2_4dof{suffix}'

    # ── CasADi Function access ────────────────────────────────────────────────

    def f_expl_fn(self):
        """ca.Function: (x, u, p) → dx/dt"""
        args = [self._x_sym, self._u_sym]
        if self._p_sym is not None:
            args.append(self._p_sym)
        return ca.Function('f_expl', args, [self._f_expl],
                           ['x', 'u'] + (['p'] if self._p_sym is not None else []),
                           ['f'])

    def f_jac_x(self):
        """∂f/∂x evaluated at a point."""
        return ca.jacobian(self._f_expl, self._x_sym)

    def f_jac_u(self):
        """∂f/∂u evaluated at a point."""
        return ca.jacobian(self._f_expl, self._u_sym)

    # ── Internal builder ──────────────────────────────────────────────────────

    def _build(self):
        vp = self.vp

        # ── Symbolic state ────────────────────────────────────────────────────
        x_pos = ca.SX.sym('x_pos')
        y_pos = ca.SX.sym('y_pos')
        z_pos = ca.SX.sym('z_pos')
        psi   = ca.SX.sym('psi')
        u_b   = ca.SX.sym('u_b')   # surge velocity
        v_b   = ca.SX.sym('v_b')   # sway velocity
        w_b   = ca.SX.sym('w_b')   # heave velocity
        r_b   = ca.SX.sym('r_b')   # yaw rate

        x_sym = ca.vertcat(x_pos, y_pos, z_pos, psi, u_b, v_b, w_b, r_b)

        # ── Symbolic control ──────────────────────────────────────────────────
        Fx  = ca.SX.sym('Fx')    # surge force  [N]
        Fy  = ca.SX.sym('Fy')    # sway  force  [N]
        Fz  = ca.SX.sym('Fz')    # heave force  [N]
        Mz  = ca.SX.sym('Mz')    # yaw  torque  [N·m]
        u_sym = ca.vertcat(Fx, Fy, Fz, Mz)

        # ── Symbolic disturbance parameter ────────────────────────────────────
        if self.with_disturbance:
            p_sym = ca.SX.sym('p', self.nx)
        else:
            p_sym = None

        # ── Effective inertia denominators ────────────────────────────────────
        Mu  = vp.mass - vp.X_ud      # surge   effective mass
        Mv  = vp.mass - vp.Y_vd      # sway    effective mass
        Mw  = vp.mass - vp.Z_wd      # heave   effective mass
        Jr  = vp.I_zz - vp.N_rd      # yaw     effective inertia

        # ── Kinematics ────────────────────────────────────────────────────────
        dx    = ca.cos(psi) * u_b - ca.sin(psi) * v_b
        dy    = ca.sin(psi) * u_b + ca.cos(psi) * v_b
        dz    = w_b
        dpsi  = r_b

        # ── Hydrodynamic drag (quadratic + linear) ────────────────────────────
        #   ca.fabs() is C1 and differentiable: avoids sign() discontinuity
        drag_u = (vp.X_u + vp.X_uc * ca.fabs(u_b)) * u_b
        drag_v = (vp.Y_v + vp.Y_vc * ca.fabs(v_b)) * v_b
        drag_w = (vp.Z_w + vp.Z_wc * ca.fabs(w_b)) * w_b
        drag_r = (vp.N_r + vp.N_rc * ca.fabs(r_b)) * r_b

        # ── Coriolis / centripetal ─────────────────────────────────────────────
        cor_u  =  (vp.mass - vp.Y_vd) * v_b * r_b
        cor_v  = -(vp.mass - vp.X_ud) * u_b * r_b
        cor_r  = -(vp.X_ud - vp.Y_vd) * u_b * v_b

        # ── Gravity + buoyancy (heave) ─────────────────────────────────────────
        # Convention (matches numpy bluerov_sim.py exactly):
        #   heave eom numerator = Fz + drag_w + m*g - F_buoy
        # F_buoy > m*g (slightly buoyant) → net term is negative → vehicle
        # tends to rise (negative w_dot at rest with Fz=0).
        net_buoy = vp.mass * vp.g - vp.F_buoy   # < 0 for positively buoyant

        # ── Equations of motion ───────────────────────────────────────────────
        du = (Fx + cor_u + drag_u)       / Mu
        dv = (Fy + cor_v + drag_v)       / Mv
        dw = (Fz + drag_w + net_buoy)    / Mw   # + net_buoy (not minus)
        dr = (Mz + cor_r + drag_r)       / Jr

        f_ana = ca.vertcat(dx, dy, dz, dpsi, du, dv, dw, dr)

        # ── Add disturbance ───────────────────────────────────────────────────
        if self.with_disturbance:
            f_expl = f_ana + p_sym
        else:
            f_expl = f_ana

        # Cache
        self._x_sym  = x_sym
        self._u_sym  = u_sym
        self._p_sym  = p_sym
        self._f_expl = f_expl


def build_casadi_integrator(model: BlueROVCasadiModel,
                             dt: float,
                             method: str = 'rk',
                             n_finite_elements: int = 1) -> ca.Function:
    """
    Build a discrete-time CasADi integrator for simulation/validation.

    method            : 'rk'  — single-step 4-stage RK.
                                 With n_finite_elements=1 (default) this is
                                 mathematically identical to the numpy rk4()
                                 in bluerov_sim.py, so they can be compared
                                 directly in tests.
                        'cvodes' — SUNDIALS adaptive solver (higher accuracy).
    n_finite_elements : number of RK sub-steps within each dt interval.
                        1 → matches numpy rk4() exactly (good for validation).
                        4 → higher accuracy (good for open-loop rollout).

    Returns ca.Function: (x0, u, p) → x_next
    """
    x, u, p, f = model.get_casadi_model()
    ode = {'x': x, 'u': u if p is None else ca.vertcat(u, p), 'ode': f}
    opts = {'tf': dt, 'simplify': True,
            'number_of_finite_elements': n_finite_elements}
    F = ca.integrator('F', method, ode, opts)

    if p is not None:
        x0  = ca.SX.sym('x0',  model.nx)
        u0  = ca.SX.sym('u0',  model.nu)
        p0  = ca.SX.sym('p0',  model.np)
        xnext = F(x0=x0, u=ca.vertcat(u0, p0))['xf']
        return ca.Function('step', [x0, u0, p0], [xnext], ['x', 'u', 'p'], ['xn'])
    else:
        x0  = ca.SX.sym('x0',  model.nx)
        u0  = ca.SX.sym('u0',  model.nu)
        xnext = F(x0=x0, u=u0)['xf']
        return ca.Function('step', [x0, u0], [xnext], ['x', 'u'], ['xn'])
