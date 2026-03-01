"""
Reference Trajectory Generators for BlueROV2

Trajectories produce a sequence of desired states:
    [x_ref, y_ref, z_ref, psi_ref]  at each timestep.
"""

import numpy as np


def lemniscate(t: np.ndarray, a: float = 3.0, b: float = 1.5,
               z_depth: float = -2.0, omega: float = 0.3) -> np.ndarray:
    """
    Bernoulli lemniscate (figure-8) trajectory in the horizontal plane.

    Parametric form:
        x(t) = a * sin(ω t)
        y(t) = b * sin(ω t) cos(ω t)  =  b/2 * sin(2ω t)

    Heading ψ is aligned with the instantaneous velocity vector.

    Parameters
    ----------
    t      : time array (N,)
    a, b   : semi-axes (m)
    z_depth: constant depth (negative = below surface in NED)
    omega  : angular frequency (rad/s)

    Returns
    -------
    ref : (N, 4)  columns = [x_ref, y_ref, z_ref, psi_ref]
    """
    phase = omega * t
    x_ref = a * np.sin(phase)
    y_ref = b * np.sin(phase) * np.cos(phase)          # = b/2 * sin(2*phase)
    z_ref = z_depth * np.ones_like(t)

    # Velocity direction for heading
    dx = a * omega * np.cos(phase)
    dy = b * omega * (np.cos(phase)**2 - np.sin(phase)**2)  # derivative of b*sin*cos
    psi_ref = np.arctan2(dy, dx)

    return np.column_stack([x_ref, y_ref, z_ref, psi_ref])


def circular(t: np.ndarray, radius: float = 3.0,
             z_depth: float = -2.0, omega: float = 0.3) -> np.ndarray:
    """Circular trajectory in horizontal plane."""
    x_ref   = radius * np.cos(omega * t)
    y_ref   = radius * np.sin(omega * t)
    z_ref   = z_depth * np.ones_like(t)
    psi_ref = omega * t + np.pi / 2
    return np.column_stack([x_ref, y_ref, z_ref, psi_ref])


def setpoint(t: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Constant setpoint trajectory."""
    if target is None:
        target = np.array([3.0, 3.0, -2.0, 0.0])
    return np.tile(target, (len(t), 1))


def helix(t: np.ndarray, radius: float = 3.0,
          descent_rate: float = 0.1, omega: float = 0.3) -> np.ndarray:
    """Helical trajectory (horizontal circle with depth change)."""
    x_ref   = radius * np.cos(omega * t)
    y_ref   = radius * np.sin(omega * t)
    z_ref   = -descent_rate * t
    psi_ref = omega * t + np.pi / 2
    return np.column_stack([x_ref, y_ref, z_ref, psi_ref])


def get_trajectory(name: str, t: np.ndarray, **kwargs) -> np.ndarray:
    """Factory function for trajectory selection."""
    registry = {
        'lemniscate': lemniscate,
        'circular':   circular,
        'setpoint':   setpoint,
        'helix':      helix,
    }
    if name not in registry:
        raise ValueError(f"Unknown trajectory '{name}'. "
                         f"Available: {list(registry.keys())}")
    return registry[name](t, **kwargs)
