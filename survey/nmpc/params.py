"""
NMPC hyperparameters for the BlueROV2 4-DOF AUV.

All tuneable parameters are collected here for clarity.  Import this
dataclass wherever the NMPC is built or called — never scatter magic
numbers through the code.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class VehicleParams:
    """Physical parameters of the BlueROV2 Heavy (SI units)."""
    # Rigid-body mass and buoyancy
    mass:    float = 11.4       # kg
    g:       float = 9.82       # m/s²
    F_buoy:  float = 115.55     # N   (1026 * 0.0115 * 9.82)

    # Added mass (SNAME notation, negative values)
    X_ud: float = -2.6
    Y_vd: float = -18.5
    Z_wd: float = -13.3
    N_rd: float = -0.28

    # Moment of inertia
    I_zz: float = 0.245         # kg·m²

    # Linear damping
    X_u: float = -0.09
    Y_v: float = -0.26
    Z_w: float = -0.19
    N_r: float = -4.64

    # Quadratic damping
    X_uc: float = -34.96
    Y_vc: float = -103.25
    Z_wc: float = -74.23
    N_rc: float = -0.43

    # Actuator limits
    F_max: float = 40.0         # N   (surge / sway / heave)
    M_max: float = 10.0         # N·m (yaw torque)


@dataclass
class NMPCParams:
    """
    Acados NMPC hyperparameters for the BlueROV2 AUV.

    Attributes
    ----------
    N_horizon : int
        Prediction horizon (number of steps).
    dt : float
        Sampling interval [s].
    T_horizon : float
        Total prediction horizon time [s] = N_horizon * dt.

    Stage cost weights (W diagonal)
    --------------------------------
    Q_x, Q_y, Q_z   : position tracking weights
    Q_psi            : heading tracking weight
    R_X, R_Y, R_Z   : force effort weights
    R_Mz             : torque effort weight

    Terminal cost weights (W_e diagonal)
    -------------------------------------
    Q_x_e, Q_y_e, Q_z_e, Q_psi_e : terminal position / heading

    Solver options
    --------------
    qp_solver         : HPIPM QP solver variant
    integrator_type   : ERK (explicit Runge-Kutta)
    nlp_solver_type   : SQP or SQP_RTI (real-time iteration)
    nlp_max_iter      : SQP outer iterations
    qp_max_iter       : HPIPM inner iterations
    levenberg_marquardt : LM regularisation (aids convergence near singularities)
    """
    # Horizon
    N_horizon: int   = 20
    dt:        float = 0.05     # s

    # ── Stage cost weights ───────────────────────────────────────────────────
    Q_x:   float = 50.0
    Q_y:   float = 50.0
    Q_z:   float = 50.0
    Q_psi: float = 20.0

    R_X:   float = 0.005
    R_Y:   float = 0.005
    R_Z:   float = 0.005
    R_Mz:  float = 0.05

    # ── Terminal cost weights ─────────────────────────────────────────────────
    Q_x_e:   float = 100.0
    Q_y_e:   float = 100.0
    Q_z_e:   float = 100.0
    Q_psi_e: float = 40.0

    # ── Solver options ────────────────────────────────────────────────────────
    qp_solver:           str   = 'PARTIAL_CONDENSING_HPIPM'
    integrator_type:     str   = 'ERK'
    nlp_solver_type:     str   = 'SQP_RTI'   # real-time iteration
    nlp_max_iter:        int   = 1            # RTI: single SQP step per call
    qp_max_iter:         int   = 50
    levenberg_marquardt: float = 1e-3

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def T_horizon(self) -> float:
        return self.N_horizon * self.dt

    @property
    def W(self) -> np.ndarray:
        """Stage cost weight matrix (ny × ny)."""
        diag = [self.Q_x, self.Q_y, self.Q_z, self.Q_psi,
                self.R_X, self.R_Y, self.R_Z, self.R_Mz]
        return np.diag(diag).astype(np.float64)

    @property
    def W_e(self) -> np.ndarray:
        """Terminal cost weight matrix (ny_e × ny_e)."""
        diag = [self.Q_x_e, self.Q_y_e, self.Q_z_e, self.Q_psi_e]
        return np.diag(diag).astype(np.float64)
