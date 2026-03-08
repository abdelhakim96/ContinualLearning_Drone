"""Simple shooting MPC using a learned dynamics model.

At each time step, optimises a sequence of controls over a short horizon
by random shooting (sample N random action sequences, evaluate cost,
pick the best). This is model-agnostic — it works with any BaseModel.
"""
import numpy as np
from survey.src.bluerov_sim import ssa
from survey.src.parameters import MAX_FORCE, MAX_TORQUE


class MPCController:
    """Random-shooting MPC with configurable dynamics model."""

    def __init__(self, model, horizon=10, n_samples=200, dt=0.05,
                 Q_pos=10.0, Q_head=5.0, R=0.01):
        self.model = model
        self.H = horizon
        self.N = n_samples
        self.dt = dt
        self.Q_pos = Q_pos
        self.Q_head = Q_head
        self.R = R
        self._u_lim = np.array([MAX_FORCE, MAX_FORCE, MAX_FORCE, MAX_TORQUE])

    def control(self, x, ref_seq):
        """
        Parameters
        ----------
        x       : current state (8,)
        ref_seq : reference trajectory (H, 4)  [x, y, z, psi]

        Returns
        -------
        u_best : best first control action (4,)
        """
        H = min(self.H, len(ref_seq))
        # Sample random action sequences
        U_samples = np.random.uniform(
            -self._u_lim, self._u_lim, size=(self.N, H, 4))

        # Add a zero-control candidate
        U_samples[0] = 0.0

        costs = np.full(self.N, np.inf)
        for i in range(self.N):
            cost = 0.0
            xi = x.copy()
            valid = True
            for t in range(H):
                try:
                    xi = self.model.predict(xi, U_samples[i, t])
                except Exception:
                    valid = False
                    break
                if not np.all(np.isfinite(xi)):
                    valid = False
                    break
                # Position cost
                dp = xi[:3] - ref_seq[t, :3]
                cost += self.Q_pos * np.dot(dp, dp)
                # Heading cost
                dh = ssa(xi[3] - ref_seq[t, 3])
                cost += self.Q_head * dh ** 2
                # Control cost
                cost += self.R * np.dot(U_samples[i, t], U_samples[i, t])
            if valid:
                costs[i] = cost

        best = np.argmin(costs)
        return np.clip(U_samples[best, 0],
                       -self._u_lim, self._u_lim)
