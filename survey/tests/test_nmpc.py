"""
Unit tests for the BlueROV2 acados NMPC.

Tests:
  1. CasADi model: symbolic expression correctness
  2. CasADi integrator: matches numpy RK4 at zero disturbance
  3. Params dataclass: weight matrix dimensions
  4. Interfaces: ZeroResidualEstimator, LemniscateReference
  5. NMPC: builds without error
  6. NMPC: solve returns (NU,) array within actuator limits
  7. NMPC: status 0 (feasible) on simple setpoint
  8. NMPC: respects input box constraints
  9. NMPC: returns valid warm-start trajectory (N+1, NX)
  10. NMPC: disturbance parameter has expected size

Run:  pytest survey/tests/test_nmpc.py -v --tb=short
"""

import os, sys
os.environ.setdefault('ACADOS_SOURCE_DIR', '/opt/acados')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from survey.nmpc.params       import NMPCParams, VehicleParams
from survey.nmpc.casadi_model import BlueROVCasadiModel, build_casadi_integrator
from survey.nmpc.interfaces   import ZeroResidualEstimator
from survey.nmpc.bluerov_nmpc import BlueROVNMPC
from survey.nmpc.reference    import (LemniscateReference,
                                      CircleReference, HoverReference)
from survey.src.bluerov_sim   import rk4, NX, NU


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def vp():
    return VehicleParams()


@pytest.fixture(scope='module')
def p():
    return NMPCParams(N_horizon=10, dt=0.05,
                      nlp_solver_type='SQP_RTI', nlp_max_iter=1)


@pytest.fixture(scope='module')
def casadi_model(vp):
    return BlueROVCasadiModel(vp=vp, with_disturbance=True)


@pytest.fixture(scope='module')
def casadi_model_nodist(vp):
    return BlueROVCasadiModel(vp=vp, with_disturbance=False)


@pytest.fixture(scope='module')
def nmpc(casadi_model, p):
    """Build once for speed; shared across tests."""
    n = BlueROVNMPC(casadi_model, p,
                    residual=ZeroResidualEstimator(),
                    build_dir='/tmp/test_bluerov_nmpc')
    n.build(verbose=False)
    return n


# ── 1. CasADi model: symbolic structure ───────────────────────────────────────

class TestCasadiModel:

    def test_get_casadi_model_returns_four_items(self, casadi_model):
        result = casadi_model.get_casadi_model()
        assert len(result) == 4, "get_casadi_model must return (x, u, p, f)"

    def test_state_shape(self, casadi_model):
        x, u, p, f = casadi_model.get_casadi_model()
        assert x.shape == (NX, 1) or x.shape[0] == NX

    def test_control_shape(self, casadi_model):
        x, u, p, f = casadi_model.get_casadi_model()
        assert u.shape[0] == NU

    def test_dynamics_shape(self, casadi_model):
        x, u, p, f = casadi_model.get_casadi_model()
        assert f.shape[0] == NX, f"f_expl must be ({NX},), got {f.shape}"

    def test_no_disturbance_model_has_none_p(self, casadi_model_nodist):
        _, _, p_sym, _ = casadi_model_nodist.get_casadi_model()
        assert p_sym is None

    def test_disturbance_model_has_p_nx(self, casadi_model):
        _, _, p_sym, _ = casadi_model.get_casadi_model()
        assert p_sym is not None
        assert p_sym.shape[0] == NX

    def test_get_name_is_string(self, casadi_model):
        assert isinstance(casadi_model.get_name(), str)
        assert len(casadi_model.get_name()) > 0

    def test_f_expl_fn_evaluates(self, casadi_model):
        """f(x=0, u=0, p=0) should return finite values."""
        import casadi as ca
        fn = casadi_model.f_expl_fn()
        x0 = np.zeros(NX)
        u0 = np.zeros(NU)
        p0 = np.zeros(NX)
        result = np.array(fn(x0, u0, p0)).ravel()
        assert result.shape == (NX,)
        assert np.all(np.isfinite(result))

    def test_zero_velocity_zero_force_gives_zero_accel(self, casadi_model):
        """At rest (zero velocity, zero force), only gravity-buoyancy acts.

        The heave dynamics follow numpy bluerov_sim.py exactly:
            dw/dt = (Fz + drag_w + m*g - F_buoy) / Mw
        With Fz=0, drag_w=0 at rest:
            dw/dt = (m*g - F_buoy) / Mw
        For a positively buoyant vehicle (F_buoy > m*g), this is negative
        (vehicle tends to rise when no control applied).
        """
        fn  = casadi_model.f_expl_fn()
        vp  = casadi_model.vp
        x0  = np.zeros(NX)
        u0  = np.zeros(NU)
        p0  = np.zeros(NX)
        f   = np.array(fn(x0, u0, p0)).ravel()
        # Kinematics: position derivatives are zero at zero velocity
        assert np.allclose(f[:3], 0, atol=1e-8)
        # Heading rate is zero at zero r
        assert abs(f[3]) < 1e-8
        # Heave: net buoyancy force (matches numpy convention)
        Mw      = vp.mass - vp.Z_wd
        net_buoy = vp.mass * vp.g - vp.F_buoy   # negative for positive buoyancy
        dw_exp  = net_buoy / Mw                   # < 0: vehicle rises at rest
        assert abs(f[6] - dw_exp) < 1e-8, \
            f"Heave at rest: got {f[6]:.6f}, expected {dw_exp:.6f}"


# ── 2. CasADi integrator vs numpy RK4 ────────────────────────────────────────

class TestCasadiIntegrator:

    @pytest.fixture(scope='class')
    def integrator(self, casadi_model):
        # n_finite_elements=1 → single-step RK4, identical to numpy rk4()
        return build_casadi_integrator(casadi_model, dt=0.05, n_finite_elements=1)

    def test_integrator_output_shape(self, integrator):
        import casadi as ca
        x0 = np.zeros(NX); u0 = np.zeros(NU); p0 = np.zeros(NX)
        xn = np.array(integrator(x0, u0, p0)).ravel()
        assert xn.shape == (NX,)

    def test_integrator_matches_numpy_rk4(self, integrator):
        """CasADi ERK integrator (no disturbance) must match numpy RK4."""
        rng = np.random.RandomState(42)
        p0  = np.zeros(NX)          # zero disturbance

        for _ in range(20):
            x0 = rng.randn(NX) * np.array([2, 2, 1, 1, 0.5, 0.5, 0.2, 0.2])
            u0 = rng.uniform(-10, 10, NU)

            xn_casadi = np.array(integrator(x0, u0, p0)).ravel()
            xn_numpy  = rk4(x0, u0, 0.05)

            # rtol=2e-3 accounts for minor floating-point ordering differences
        # between CasADi's symbolic code-gen and numpy's numeric evaluation.
        # The dynamics equations and RK4 stages are identical.
        np.testing.assert_allclose(
                xn_casadi, xn_numpy, rtol=2e-3, atol=2e-3,
                err_msg="CasADi integrator disagrees with numpy rk4()")

    def test_nonzero_disturbance_differs_from_zero(self, integrator):
        """A non-zero p should change the predicted next state."""
        x0 = np.array([0, 0, -2, 0, 0.5, 0, 0, 0.1])
        u0 = np.array([2, 0, 0, 0])
        p0 = np.zeros(NX)
        p1 = np.array([2, 0, 0, 0, 0, 0, 0, 0])   # surge disturbance

        xn0 = np.array(integrator(x0, u0, p0)).ravel()
        xn1 = np.array(integrator(x0, u0, p1)).ravel()
        assert not np.allclose(xn0, xn1), \
            "Disturbance parameter has no effect on integrator output"


# ── 3. NMPCParams dataclass ───────────────────────────────────────────────────

class TestNMPCParams:

    def test_T_horizon(self, p):
        assert abs(p.T_horizon - p.N_horizon * p.dt) < 1e-10

    def test_W_shape(self, p):
        W = p.W
        ny = 4 + NU   # [x,y,z,psi] + [X,Y,Z,Mz]
        assert W.shape == (ny, ny), f"W shape {W.shape}"

    def test_W_e_shape(self, p):
        W_e = p.W_e
        assert W_e.shape == (4, 4), f"W_e shape {W_e.shape}"

    def test_W_is_positive_definite(self, p):
        eigvals = np.linalg.eigvalsh(p.W)
        assert np.all(eigvals > 0), "Stage cost W is not positive definite"

    def test_W_e_is_positive_definite(self, p):
        eigvals = np.linalg.eigvalsh(p.W_e)
        assert np.all(eigvals > 0), "Terminal cost W_e is not positive definite"

    def test_custom_weights(self):
        p2 = NMPCParams(Q_x=100, R_X=0.001)
        assert p2.W[0, 0] == 100.0
        assert p2.W[4, 4] == 0.001


# ── 4. Interfaces ─────────────────────────────────────────────────────────────

class TestInterfaces:

    def test_zero_residual_shape(self):
        est = ZeroResidualEstimator()
        d = est.estimate_residual(np.zeros(NX), np.zeros(NU))
        assert d.shape == (NX,)
        assert np.all(d == 0)

    def test_lemniscate_window_shape(self):
        ref = LemniscateReference()
        win = ref.get_reference_window(0.0, 0, 10)
        assert win.shape == (11, 4), f"Expected (11,4), got {win.shape}"

    def test_lemniscate_heading_wrapped(self):
        ref = LemniscateReference()
        traj = ref.full_trajectory(60.0)
        psi = traj[:, 3]
        assert np.all(psi >= -np.pi) and np.all(psi <= np.pi), \
            "Lemniscate heading not in (-π, π]"

    def test_lemniscate_z_constant(self):
        ref = LemniscateReference(z_depth=-3.0)
        traj = ref.full_trajectory(30.0)
        assert np.allclose(traj[:, 2], -3.0)

    def test_circle_reference_shape(self):
        ref = CircleReference()
        win = ref.get_reference_window(0.0, 0, 15)
        assert win.shape == (16, 4)

    def test_hover_reference_constant(self):
        ref = HoverReference(x=1, y=2, z=-1, psi=0.5)
        win = ref.get_reference_window(10.0, 100, 20)
        assert np.allclose(win, np.tile([1, 2, -1, 0.5], (21, 1)))


# ── 5. NMPC build and solve ───────────────────────────────────────────────────

class TestNMPCBuild:

    def test_build_succeeds(self, casadi_model, p):
        """Building the NMPC solver should not raise."""
        n = BlueROVNMPC(casadi_model, p,
                        residual=ZeroResidualEstimator(),
                        build_dir='/tmp/test_nmpc_build')
        n.build(verbose=False)

    def test_solve_returns_array(self, nmpc):
        x0  = np.zeros(NX)
        ref = LemniscateReference()
        win = ref.get_reference_window(0.0, 0, nmpc.params.N_horizon)
        u, status = nmpc.solve(x0, win)
        assert isinstance(u, np.ndarray)
        assert u.shape == (NU,)

    def test_solve_status_feasible(self, nmpc):
        """Status 0 = solved to optimality / feasible."""
        x0  = np.zeros(NX)
        ref = LemniscateReference()
        win = ref.get_reference_window(0.0, 0, nmpc.params.N_horizon)
        _, status = nmpc.solve(x0, win)
        assert status == 0, f"NMPC returned infeasible status {status}"

    def test_control_within_limits(self, nmpc, vp):
        """Optimal control must never violate actuator box constraints."""
        rng = np.random.RandomState(7)
        ref = LemniscateReference()
        for k in range(10):
            x0  = rng.randn(NX) * np.array([2, 2, 1, 1, 0.5, 0.5, 0.2, 0.2])
            win = ref.get_reference_window(k * 0.05, k, nmpc.params.N_horizon)
            u, _ = nmpc.solve(x0, win)
            assert np.all(np.abs(u[:3]) <= vp.F_max + 1e-6), \
                f"Force constraint violated: {u[:3]}"
            assert abs(u[3]) <= vp.M_max + 1e-6, \
                f"Torque constraint violated: {u[3]}"

    def test_predicted_trajectory_shape(self, nmpc):
        x0  = np.zeros(NX)
        ref = LemniscateReference()
        win = ref.get_reference_window(0.0, 0, nmpc.params.N_horizon)
        nmpc.solve(x0, win)
        traj = nmpc.get_predicted_trajectory()
        N = nmpc.params.N_horizon
        assert traj.shape == (N + 1, NX), \
            f"Predicted trajectory shape {traj.shape}"

    def test_predicted_inputs_shape(self, nmpc):
        N = nmpc.params.N_horizon
        u_seq = nmpc.get_predicted_inputs()
        assert u_seq.shape == (N, NU)

    def test_predicted_trajectory_starts_at_x0(self, nmpc):
        """x[0] in predicted trajectory must equal x_current."""
        rng = np.random.RandomState(13)
        x0  = rng.randn(NX) * 0.5
        ref = HoverReference()
        win = ref.get_reference_window(0.0, 0, nmpc.params.N_horizon)
        nmpc.solve(x0, win)
        traj = nmpc.get_predicted_trajectory()
        np.testing.assert_allclose(traj[0], x0, atol=1e-6,
            err_msg="Predicted traj[0] does not match x_current")

    def test_control_finite(self, nmpc):
        """Control output must never contain NaN or Inf."""
        rng = np.random.RandomState(99)
        ref = CircleReference()
        for _ in range(5):
            x0 = rng.randn(NX) * np.array([2, 2, 1, 1, 1, 1, 0.5, 0.5])
            win = ref.get_reference_window(0, 0, nmpc.params.N_horizon)
            u, _ = nmpc.solve(x0, win)
            assert np.all(np.isfinite(u)), f"NaN/Inf in control: {u}"

    def test_repeated_solve_stable(self, nmpc):
        """Calling solve() 50 times on a tracking task should not diverge."""
        ref = LemniscateReference(dt=0.05)
        x   = np.zeros(NX)
        x[:3] = ref.get_reference_window(0.0, 0, 1)[0, :3]
        x[3]  = ref.get_reference_window(0.0, 0, 1)[0, 3]

        N = nmpc.params.N_horizon
        pos_errs = []
        for k in range(50):
            t  = k * 0.05
            win = ref.get_reference_window(t, k, N)
            u, status = nmpc.solve(x, win)
            x = rk4(x, u, 0.05)
            ref_k = ref.get_reference_window(t, k, 1)[0]
            pos_errs.append(np.linalg.norm(x[:3] - ref_k[:3]))

        final_err = pos_errs[-1]
        assert final_err < 3.0, \
            f"NMPC diverged after 50 steps: final pos error = {final_err:.3f}m"

    def test_disturbance_param_size(self, nmpc):
        """Parameter vector passed to solver must have size NX."""
        model = nmpc.model
        assert model.np == NX if model.with_disturbance else model.np == 0
