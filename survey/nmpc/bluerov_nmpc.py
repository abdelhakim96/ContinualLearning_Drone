"""
BlueROV2 Nonlinear MPC using acados.

Architecture
────────────
  BlueROVNMPC
    ├── BlueROVCasadiModel   — symbolic dynamics (CasADi / acados)
    ├── NMPCParams           — all tuning knobs in one dataclass
    ├── ResidualEstimator    — optional learned disturbance (plug-in)
    └── ReferenceProvider    — desired trajectory window

Usage
─────
  from survey.nmpc.params          import NMPCParams, VehicleParams
  from survey.nmpc.casadi_model    import BlueROVCasadiModel
  from survey.nmpc.bluerov_nmpc    import BlueROVNMPC
  from survey.nmpc.interfaces      import ZeroResidualEstimator

  model  = BlueROVCasadiModel()
  params = NMPCParams()
  nmpc   = BlueROVNMPC(model, params, residual=ZeroResidualEstimator())
  nmpc.build()                          # compile acados solver (once)

  u_opt = nmpc.solve(x_current, ref_window)   # ref_window shape (N+1, 4)
"""

import os
import tempfile
import shutil
import numpy as np
import casadi as ca

os.environ.setdefault('ACADOS_SOURCE_DIR', '/opt/acados')

from acados_template import (AcadosModel, AcadosOcp,
                              AcadosOcpSolver, AcadosSimSolver)

from .params       import NMPCParams, VehicleParams
from .casadi_model import BlueROVCasadiModel
from .interfaces   import ResidualEstimator, ZeroResidualEstimator


# ── State / input dimension constants ────────────────────────────────────────
NX   = 8    # [x,y,z,psi,u,v,w,r]
NU   = 4    # [X,Y,Z,Mz]
NY   = NX // 2 + NU   # stage output = [x,y,z,psi, X,Y,Z,Mz]  (= 4+4 = 8)
NY_E = 4              # terminal output = [x,y,z,psi]

# State indices
IDX_POS  = slice(0, 3)   # x,y,z
IDX_PSI  = 3
IDX_VEL  = slice(4, 8)   # u,v,w,r


class BlueROVNMPC:
    """
    Nonlinear MPC for BlueROV2 trajectory tracking.

    Parameters
    ----------
    model     : BlueROVCasadiModel  — symbolic dynamics
    params    : NMPCParams          — horizon, weights, solver settings
    residual  : ResidualEstimator   — optional learned disturbance
    build_dir : str or None         — directory for generated C code
                                      (None → system temp dir)
    """

    def __init__(self,
                 model:    BlueROVCasadiModel = None,
                 params:   NMPCParams         = None,
                 residual: ResidualEstimator   = None,
                 build_dir: str = None):
        self.model    = model   or BlueROVCasadiModel()
        self.params   = params  or NMPCParams()
        self.residual = residual or ZeroResidualEstimator()
        self._build_dir   = build_dir
        self._solver      = None
        self._built       = False
        self._last_u      = np.zeros(NU)

    # ── Build (compile acados solver) ─────────────────────────────────────────

    def build(self, verbose: bool = False) -> 'BlueROVNMPC':
        """
        Generate, compile, and load the acados OCP solver.
        Must be called once before solve().
        Returns self for chaining: nmpc.build().solve(...)
        """
        p = self.params
        m = self.model
        vp = m.vp

        # ── CasADi model ──────────────────────────────────────────────────────
        x_sym, u_sym, p_sym, f_expl = m.get_casadi_model()

        # ── Acados model ──────────────────────────────────────────────────────
        acados_model = AcadosModel()
        acados_model.name = m.get_name()
        acados_model.x    = x_sym
        acados_model.u    = u_sym
        acados_model.f_expl_expr = f_expl

        # Implicit form (required for some integrators)
        xdot = ca.SX.sym('xdot', NX)
        acados_model.xdot         = xdot
        acados_model.f_impl_expr  = xdot - f_expl

        if p_sym is not None:
            acados_model.p = p_sym

        # ── Stage cost output:  y = [x[0..3], u[0..3]] ───────────────────────
        #   Tracking [x,y,z,psi] + effort [X,Y,Z,Mz]
        acados_model.cost_y_expr   = ca.vertcat(x_sym[:4], u_sym)
        acados_model.cost_y_expr_e = x_sym[:4]

        # ── OCP setup ─────────────────────────────────────────────────────────
        ocp = AcadosOcp()
        ocp.model = acados_model

        # Horizon
        ocp.solver_options.N_horizon = p.N_horizon
        ocp.solver_options.tf        = p.T_horizon

        # ── Cost ──────────────────────────────────────────────────────────────
        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W   = p.W      # (ny, ny)
        ocp.cost.W_e = p.W_e    # (ny_e, ny_e)

        # Initial reference values (will be overwritten every solve() call)
        ocp.cost.yref   = np.zeros(NY)
        ocp.cost.yref_e = np.zeros(NY_E)

        # ── Constraints ───────────────────────────────────────────────────────
        # Input box constraints: [X, Y, Z, Mz]
        ocp.constraints.lbu = np.array([-vp.F_max, -vp.F_max,
                                        -vp.F_max, -vp.M_max])
        ocp.constraints.ubu = np.array([ vp.F_max,  vp.F_max,
                                          vp.F_max,  vp.M_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Initial state (placeholder — updated before each solve)
        ocp.constraints.x0 = np.zeros(NX)

        # ── Parameter initialisation (disturbance = 0 initially) ──────────────
        if p_sym is not None:
            ocp.parameter_values = np.zeros(m.np)

        # ── Solver options ────────────────────────────────────────────────────
        ocp.solver_options.qp_solver         = p.qp_solver
        ocp.solver_options.integrator_type   = p.integrator_type
        ocp.solver_options.nlp_solver_type   = p.nlp_solver_type
        ocp.solver_options.nlp_solver_max_iter = p.nlp_max_iter
        ocp.solver_options.qp_solver_iter_max  = p.qp_max_iter
        ocp.solver_options.levenberg_marquardt = p.levenberg_marquardt
        ocp.solver_options.print_level         = 0

        # Use a dedicated temp/build directory so generated code doesn't
        # pollute the working directory
        if self._build_dir is None:
            self._build_dir = tempfile.mkdtemp(prefix='bluerov_nmpc_')
        os.makedirs(self._build_dir, exist_ok=True)

        json_file = os.path.join(self._build_dir, f'{m.get_name()}_ocp.json')

        self._solver = AcadosOcpSolver(ocp,
                                        json_file=json_file,
                                        verbose=verbose)
        self._built = True
        return self

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self,
              x_current: np.ndarray,
              ref_window: np.ndarray) -> np.ndarray:
        """
        Run one NMPC iteration.

        Parameters
        ----------
        x_current  : current state (NX,) = [x,y,z,psi,u,v,w,r]
        ref_window : reference trajectory (N+1, 4) = [x_d, y_d, z_d, psi_d]
                     ref_window[k] = reference at stage k
                     ref_window[N] = terminal reference

        Returns
        -------
        u_opt : optimal first control (NU,) = [X, Y, Z, Mz]
        """
        assert self._built, "Call build() before solve()."

        N = self.params.N_horizon
        s = self._solver

        # ── Set initial state constraint ─────────────────────────────────────
        s.set(0, 'lbx', x_current)
        s.set(0, 'ubx', x_current)

        # ── Set disturbance parameter ─────────────────────────────────────────
        if self.model.with_disturbance:
            d_est = self.residual.estimate_residual(x_current, self._last_u)
            for k in range(N + 1):
                s.set(k, 'p', d_est.astype(np.float64))

        # ── Set reference trajectory ──────────────────────────────────────────
        ref_window = _pad_ref(ref_window, N + 1)

        for k in range(N):
            yref_k = np.zeros(NY)
            yref_k[:4] = ref_window[k, :4]   # [x_d, y_d, z_d, psi_d, 0,0,0,0]
            s.set(k, 'yref', yref_k)

        yref_e = ref_window[N, :4].astype(np.float64)
        s.set(N, 'yref', yref_e)

        # ── Solve OCP ─────────────────────────────────────────────────────────
        status = s.solve()

        # Get optimal first input
        u_opt = s.get(0, 'u').copy()

        # Clamp to box constraints (safety belt)
        vp = self.model.vp
        u_opt[:3] = np.clip(u_opt[:3], -vp.F_max,  vp.F_max)
        u_opt[3]  = np.clip(u_opt[3],  -vp.M_max,  vp.M_max)

        self._last_u = u_opt.copy()
        return u_opt, status

    # ── Warm-start helpers ────────────────────────────────────────────────────

    def set_initial_guess(self, x_traj: np.ndarray, u_traj: np.ndarray):
        """
        Provide an initial guess for all horizon stages.
        x_traj : (N+1, NX),  u_traj : (N, NU)
        """
        N = self.params.N_horizon
        for k in range(N):
            self._solver.set(k, 'x', x_traj[k].astype(np.float64))
            self._solver.set(k, 'u', u_traj[k].astype(np.float64))
        self._solver.set(N, 'x', x_traj[N].astype(np.float64))

    def get_predicted_trajectory(self) -> np.ndarray:
        """Return the predicted state trajectory (N+1, NX) from last solve."""
        N = self.params.N_horizon
        return np.array([self._solver.get(k, 'x')
                         for k in range(N + 1)])

    def get_predicted_inputs(self) -> np.ndarray:
        """Return the predicted input sequence (N, NU) from last solve."""
        N = self.params.N_horizon
        return np.array([self._solver.get(k, 'u')
                         for k in range(N)])

    # ── Clean up generated files ──────────────────────────────────────────────

    def cleanup(self):
        """Remove generated C code and build artefacts."""
        if self._build_dir and os.path.isdir(self._build_dir):
            shutil.rmtree(self._build_dir, ignore_errors=True)


# ── Reference window helper ────────────────────────────────────────────────────

def _pad_ref(ref: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure ref has at least `target_len` rows by repeating the last row."""
    if len(ref) >= target_len:
        return ref[:target_len]
    n_pad = target_len - len(ref)
    return np.vstack([ref, np.tile(ref[-1], (n_pad, 1))])
