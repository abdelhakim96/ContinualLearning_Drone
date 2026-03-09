"""
Reference trajectory generators for the BlueROV2 NMPC.

All generators implement ReferenceProvider and return windows of shape (N+1, 4).
"""

import numpy as np
from .interfaces import ReferenceProvider


def _ssa(a):
    """Smallest signed angle: wraps to (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class LemniscateReference(ReferenceProvider):
    """
    Lemniscate (figure-8) reference trajectory.

    x(t) = a * sin(ω t)
    y(t) = b * sin(ω t) cos(ω t)
    z(t) = z_depth (constant)
    ψ(t) = atan2(dy/dt, dx/dt)

    Parameters
    ----------
    a, b    : semi-axes [m]
    omega   : angular frequency [rad/s]
    z_depth : target depth [m] (negative = downward)
    dt      : sampling interval [s]
    """

    def __init__(self,
                 a: float = 3.0,
                 b: float = 1.5,
                 omega: float = 0.25,
                 z_depth: float = -2.0,
                 dt: float = 0.05):
        self.a = a; self.b = b; self.omega = omega
        self.z_depth = z_depth; self.dt = dt

    def _eval(self, t: np.ndarray) -> np.ndarray:
        """Evaluate (N, 4) trajectory at times t."""
        om = self.omega
        x   =  self.a * np.sin(om * t)
        y   =  self.b * np.sin(om * t) * np.cos(om * t)
        z   =  self.z_depth * np.ones_like(t)
        dx  =  self.a * om * np.cos(om * t)
        dy  =  self.b * om * (np.cos(om * t)**2 - np.sin(om * t)**2)
        psi =  np.arctan2(dy, dx)
        return np.column_stack([x, y, z, psi])

    def get_reference_window(self,
                             t_current: float,
                             step: int,
                             N_horizon: int) -> np.ndarray:
        t_window = t_current + np.arange(N_horizon + 1) * self.dt
        return self._eval(t_window)

    def full_trajectory(self, T: float) -> np.ndarray:
        """Full trajectory array for plotting: (K+1, 4)."""
        t = np.arange(0, T + self.dt, self.dt)
        return self._eval(t)


class CircleReference(ReferenceProvider):
    """Horizontal circle reference."""

    def __init__(self, radius=2.0, omega=0.3, z_depth=-2.0, dt=0.05):
        self.radius = radius; self.omega = omega
        self.z_depth = z_depth; self.dt = dt

    def _eval(self, t):
        om = self.omega
        x   = self.radius * np.cos(om * t)
        y   = self.radius * np.sin(om * t)
        z   = self.z_depth * np.ones_like(t)
        psi = np.arctan2(np.cos(om * t), -np.sin(om * t))
        return np.column_stack([x, y, z, psi])

    def get_reference_window(self, t_current, step, N_horizon):
        t = t_current + np.arange(N_horizon + 1) * self.dt
        return self._eval(t)

    def full_trajectory(self, T):
        t = np.arange(0, T + self.dt, self.dt)
        return self._eval(t)


class HoverReference(ReferenceProvider):
    """Constant setpoint."""

    def __init__(self, x=0., y=0., z=-1., psi=0., dt=0.05):
        self._ref = np.array([x, y, z, psi])
        self.dt = dt

    def get_reference_window(self, t_current, step, N_horizon):
        return np.tile(self._ref, (N_horizon + 1, 1))

    def full_trajectory(self, T):
        K = int(T / self.dt) + 1
        return np.tile(self._ref, (K, 1))
