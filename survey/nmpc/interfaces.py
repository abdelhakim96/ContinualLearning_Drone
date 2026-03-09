"""
Abstract interfaces used by the NMPC module.

Three roles are separated:
  1. DynamicsModel     — provides the CasADi symbolic continuous dynamics
  2. ResidualEstimator — provides an additive disturbance estimate from data
  3. ReferenceProvider — provides the reference trajectory window

Keeping these as ABCs lets you swap in:
  - Learned residuals (PINN, NN, SINDy, GP) via ResidualEstimator
  - Any trajectory generator via ReferenceProvider
  - Modified vehicle models via DynamicsModel
without touching the NMPC core.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


# ── 1. Dynamics model (CasADi) ────────────────────────────────────────────────

class DynamicsModel(ABC):
    """
    Provides the CasADi symbolic continuous-time dynamics
        dx/dt = f(x, u, p)
    used by acados to build the OCP.

    Parameters
    ----------
    nx : int  state dimension
    nu : int  input dimension
    np : int  parameter dimension (e.g. disturbance estimate)
    """
    nx: int
    nu: int
    np: int = 0

    @abstractmethod
    def get_casadi_model(self):
        """
        Returns (x_sym, u_sym, p_sym, f_expl_expr) where:
          x_sym      : ca.SX symbol (nx,)
          u_sym      : ca.SX symbol (nu,)
          p_sym      : ca.SX symbol (np,)   — None if np==0
          f_expl_expr: ca.SX expression (nx,)  — continuous-time rhs
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Unique model name (used in generated C-code filenames)."""
        ...


# ── 2. Residual estimator (learned model integration) ─────────────────────────

class ResidualEstimator(ABC):
    """
    Interface for plugging a learned dynamics model into the NMPC as an
    additive disturbance estimate.

    At each time step, the NMPC calls estimate_residual(x, u) to obtain
    d_est ∈ R^{nx}, which is then passed to the solver as the parameter
    vector p (if the dynamics model includes a disturbance parameter).

    This cleanly decouples offline learned models (NN, PINN, SINDy, GP)
    from the real-time NMPC solver: the learned model runs in Python,
    and its output is passed as a constant-over-horizon disturbance estimate.

    Subclass this to wrap any BaseModel:
        class PINNResidualEstimator(ResidualEstimator):
            def __init__(self, pinn_model):
                self._model = pinn_model
                self._dt    = 0.05
            def estimate_residual(self, x, u):
                xn_pinn = self._model.predict(x, u)
                xn_ana  = rk4(x, u, self._dt)
                return (xn_pinn - xn_ana) / self._dt  # finite-diff to get dx/dt
    """

    @abstractmethod
    def estimate_residual(self,
                          x: np.ndarray,
                          u: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : current state (nx,)
        u : last applied control (nu,)

        Returns
        -------
        d_est : additive residual d ∈ R^{nx} representing
                f_true(x,u) - f_analytical(x,u)  (continuous-time)
        """
        ...

    def update(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray):
        """
        Optional: called after each real step to update the estimator
        (e.g. for online GP update, moving-window SINDy).
        Default: no-op.
        """

    def reset(self):
        """Optional: reset internal state (e.g. GP posterior)."""


class ZeroResidualEstimator(ResidualEstimator):
    """No residual — pure analytical NMPC (default)."""
    def estimate_residual(self, x, u):
        return np.zeros(8)


# ── 3. Reference provider ─────────────────────────────────────────────────────

class ReferenceProvider(ABC):
    """
    Provides the reference trajectory for the NMPC at each time step.

    The NMPC queries get_reference_window(t, k, N) to get the next N
    reference points.  Return shape: (N+1, 4) — [x, y, z, psi].
    """

    @abstractmethod
    def get_reference_window(self,
                             t_current: float,
                             step: int,
                             N_horizon: int) -> np.ndarray:
        """
        Parameters
        ----------
        t_current  : current simulation time [s]
        step       : current discrete time index
        N_horizon  : number of steps in the NMPC horizon

        Returns
        -------
        ref : (N_horizon+1, 4) array of [x_d, y_d, z_d, psi_d]
              ref[0]  = current reference
              ref[N]  = terminal reference
        """
        ...
