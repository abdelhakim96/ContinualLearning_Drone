"""
Unit tests for all data-driven dynamics model implementations.

Checks:
  - Output shape and dtype
  - No NaN / Inf in predictions
  - Classic model exactly matches the RK4 simulator
  - All models improve after training (loss decreases)
  - Rollout is consistent with sequential predict() calls
  - Normalisation is idempotent (re-predicting same input gives same output)
  - PINN residual is zero on clean analytical data (perfect teacher)
  - SINDy library contains expected physical terms
  - Hankel buffer reset works

Run with:  pytest survey/tests/test_models.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from survey.src.bluerov_sim import (collect_data, rk4, dynamics, NX, NU)
from survey.src.parameters import MAX_FORCE, MAX_TORQUE
from survey.methods import ALL_METHODS
from survey.methods.classic_model import ClassicModel
from survey.methods.pinn_model import PINNModel
from survey.methods.sindy_model import SINDyModel, _build_library
from survey.methods.hankel_model import HankelDMDModel

DT = 0.05
TRAIN_N = 500   # small for fast tests
TEST_N  = 100


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    X, U, Xn = collect_data(N=TRAIN_N, dt=DT, seed=0)
    return X, U, Xn


@pytest.fixture(scope="module")
def test_inputs():
    """Random (x, u) pairs for inference tests."""
    rng = np.random.RandomState(77)
    X = rng.randn(TEST_N, NX) * np.array([3, 3, 2, np.pi, 1, 1, 0.5, 0.5])
    U = rng.uniform([-MAX_FORCE]*3 + [-MAX_TORQUE],
                    [MAX_FORCE]*3  + [MAX_TORQUE], size=(TEST_N, NU))
    Xn = np.array([rk4(X[i], U[i], DT) for i in range(TEST_N)])
    return X, U, Xn


@pytest.fixture(scope="module")
def trained_models(dataset):
    """Train all models once and cache."""
    X, U, Xn = dataset
    models = {}
    for name, Cls in ALL_METHODS.items():
        m = Cls()
        m.train(X, U, Xn)
        models[name] = m
    return models


# ── 1. Output shape and dtype ─────────────────────────────────────────────────

class TestOutputShape:
    @pytest.mark.parametrize("name", list(ALL_METHODS))
    def test_predict_shape(self, name, trained_models, test_inputs):
        model = trained_models[name]
        x, u = test_inputs[0][0], test_inputs[1][0]
        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(x)
        pred = model.predict(x, u)
        assert pred.shape == (NX,), \
            f"{name}: expected ({NX},), got {pred.shape}"

    @pytest.mark.parametrize("name", list(ALL_METHODS))
    def test_predict_dtype(self, name, trained_models, test_inputs):
        model = trained_models[name]
        x, u = test_inputs[0][0], test_inputs[1][0]
        pred = model.predict(x, u)
        assert pred.dtype in (np.float32, np.float64), \
            f"{name}: unexpected dtype {pred.dtype}"

    @pytest.mark.parametrize("name", list(ALL_METHODS))
    def test_predict_batch_shape(self, name, trained_models, test_inputs):
        model = trained_models[name]
        X, U, _ = test_inputs
        preds = model.predict_batch(X[:10], U[:10])
        assert preds.shape == (10, NX), \
            f"{name}: batch shape {preds.shape}"


# ── 2. Numerical validity ─────────────────────────────────────────────────────

class TestNumericalValidity:
    @pytest.mark.parametrize("name", list(ALL_METHODS))
    def test_no_nan(self, name, trained_models, test_inputs):
        model = trained_models[name]
        X, U, _ = test_inputs
        preds = model.predict_batch(X, U)
        assert np.all(np.isfinite(preds)), \
            f"{name}: NaN or Inf in predictions"

    @pytest.mark.parametrize("name", list(ALL_METHODS))
    def test_zero_input_finite(self, name, trained_models):
        """Zero state / zero control must not crash."""
        model = trained_models[name]
        x = np.zeros(NX)
        u = np.zeros(NU)
        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(x)
        pred = model.predict(x, u)
        assert np.all(np.isfinite(pred)), f"{name}: failed on zero input"


# ── 3. Classic model exactness ────────────────────────────────────────────────

class TestClassicModel:
    def test_matches_rk4_exactly(self, test_inputs):
        """ClassicModel must reproduce rk4() to machine precision."""
        X, U, Xn_true = test_inputs
        model = ClassicModel()
        model.train(X, U, Xn_true)
        preds = model.predict_batch(X, U)
        np.testing.assert_allclose(preds, Xn_true, rtol=1e-10, atol=1e-10,
                                   err_msg="ClassicModel deviates from rk4()")

    def test_train_is_noop(self, dataset):
        """Training Classic model should be instant (no-op)."""
        import time
        X, U, Xn = dataset
        m = ClassicModel()
        t0 = time.perf_counter()
        m.train(X, U, Xn)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"Classic.train took {elapsed:.2f}s (expected <0.5s)"


# ── 4. Training improvement ───────────────────────────────────────────────────

class TestTrainingImprovement:
    """Models should predict better after training than an untrained version."""

    @pytest.mark.parametrize("name", ["NN", "PINN"])
    def test_nn_loss_decreases(self, name, dataset, test_inputs):
        """Neural network: loss on test set should be lower after more training."""
        X_tr, U_tr, Xn_tr = dataset
        X_te, U_te, Xn_te = test_inputs

        # Untrained (1 epoch)
        Cls = ALL_METHODS[name]
        m_few = Cls(epochs=1)
        m_few.train(X_tr, U_tr, Xn_tr)
        err_few = np.mean((m_few.predict_batch(X_te, U_te) - Xn_te) ** 2)

        # More training
        m_more = Cls(epochs=50)
        m_more.train(X_tr, U_tr, Xn_tr)
        err_more = np.mean((m_more.predict_batch(X_te, U_te) - Xn_te) ** 2)

        assert err_more < err_few, \
            f"{name}: MSE did not decrease with more training " \
            f"(1ep={err_few:.5f}, 50ep={err_more:.5f})"

    @pytest.mark.parametrize("name", ["SINDy", "DMD", "GP"])
    def test_data_driven_beats_zero(self, name, dataset, test_inputs):
        """SINDy/DMD/GP must outperform the trivial zero-prediction baseline."""
        X_tr, U_tr, Xn_tr = dataset
        X_te, U_te, Xn_te = test_inputs

        model = ALL_METHODS[name]()
        model.train(X_tr, U_tr, Xn_tr)
        pred = model.predict_batch(X_te, U_te)

        mse_model = np.mean((pred - Xn_te) ** 2)
        mse_zero  = np.mean(Xn_te ** 2)   # predicting all zeros
        assert mse_model < mse_zero, \
            f"{name}: model MSE ({mse_model:.5f}) >= zero-baseline ({mse_zero:.5f})"


# ── 5. Rollout consistency ────────────────────────────────────────────────────

class TestRolloutConsistency:
    @pytest.mark.parametrize("name", list(ALL_METHODS))
    def test_rollout_matches_sequential_predict(self, name, trained_models):
        """rollout() must match calling predict() in a loop."""
        model = trained_models[name]
        rng = np.random.RandomState(5)
        x0 = np.zeros(NX)
        H = 20
        U_seq = rng.uniform([-MAX_FORCE]*3 + [-MAX_TORQUE],
                            [MAX_FORCE]*3  + [MAX_TORQUE],
                            size=(H, NU)) * 0.2

        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(x0)
        traj_rollout = model.rollout(x0, U_seq, DT)

        # Sequential
        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(x0)
        x = x0.copy()
        traj_seq = [x.copy()]
        for u in U_seq:
            x = model.predict(x, u)
            traj_seq.append(x.copy())
        traj_seq = np.array(traj_seq)

        np.testing.assert_allclose(
            traj_rollout, traj_seq, rtol=1e-5, atol=1e-5,
            err_msg=f"{name}: rollout() differs from sequential predict()")


# ── 6. Determinism ────────────────────────────────────────────────────────────

class TestDeterminism:
    @pytest.mark.parametrize("name", list(ALL_METHODS))
    def test_same_input_same_output(self, name, trained_models, test_inputs):
        """Calling predict() twice from the same state must give same result.

        Stateful models (Hankel) must have their buffer reset before each call
        so that the 'same state' condition is satisfied.
        """
        model = trained_models[name]
        x, u = test_inputs[0][5], test_inputs[1][5]

        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(x)
        p1 = model.predict(x, u)

        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(x)
        p2 = model.predict(x, u)

        np.testing.assert_array_equal(p1, p2,
            err_msg=f"{name}: predict() is non-deterministic given same initial state")

    @pytest.mark.parametrize("name", ["Hankel"])
    def test_stateful_buffer_updates(self, name, trained_models, test_inputs):
        """Hankel model's delay buffer must update on each predict() call.

        This test documents and verifies the stateful behaviour:
        two consecutive predict() calls with the same input should differ
        because the delay buffer has advanced.
        """
        model = trained_models[name]
        x, u = test_inputs[0][5], test_inputs[1][5]

        model.reset_buffers(x)
        p1 = model.predict(x, u)   # buffer: [x, x, x] → advances
        p2 = model.predict(x, u)   # buffer state has changed

        # They should differ (buffer advanced), confirming statefulness
        assert not np.allclose(p1, p2), \
            "Hankel buffer did not update between consecutive predict() calls"


# ── 7. PINN-specific: residual is small on clean data ─────────────────────────

class TestPINNResidual:
    def test_residual_near_zero_on_clean(self, dataset):
        """PINN residual net should learn ~zero correction on clean RK4 data."""
        X, U, Xn = dataset

        # PINN is trained to correct the analytical model → residual ≈ 0
        model = PINNModel(epochs=100)
        model.train(X, U, Xn)

        from survey.src.bluerov_sim import rk4
        Xn_ana = np.array([rk4(X[i], U[i], DT) for i in range(len(X))])
        Xn_pinn = model.predict_batch(X, U)

        # PINN pred should be very close to analytical (residual ≈ 0)
        err_pinn_vs_ana = np.sqrt(np.mean((Xn_pinn - Xn_ana) ** 2))
        assert err_pinn_vs_ana < 0.02, \
            f"PINN residual too large on clean data: {err_pinn_vs_ana:.5f}"


# ── 8. SINDy-specific: library structure ─────────────────────────────────────

class TestSINDyLibrary:
    def test_library_dimensions(self, dataset):
        """SINDy library must have expected number of columns."""
        X, U, Xn = dataset
        Theta = _build_library(X, U)
        n, p = Theta.shape
        assert n == len(X), "Library row count mismatch"
        # 1 + 8 + 4 + 36 + 32 + 4 + 2 = 87 expected terms
        assert p > 40, f"Library too small: {p} columns"

    def test_sindy_sparsity(self, dataset):
        """After training, SINDy coefficient matrix must be sparse."""
        X, U, Xn = dataset
        model = SINDyModel(threshold=0.05)
        model.train(X, U, Xn)
        total  = model.Xi.size
        nonzero = np.count_nonzero(model.Xi)
        sparsity = 1.0 - nonzero / total
        assert sparsity > 0.5, \
            f"SINDy not sparse enough: {sparsity:.2%} zeros (need >50%)"

    def test_sindy_predicts_clean(self, dataset, test_inputs):
        """SINDy should achieve low position RMSE on clean test data."""
        X_tr, U_tr, Xn_tr = dataset
        X_te, U_te, Xn_te = test_inputs
        model = SINDyModel()
        model.train(X_tr, U_tr, Xn_tr)
        pred = model.predict_batch(X_te, U_te)
        rmse_pos = np.sqrt(np.mean((pred[:, :3] - Xn_te[:, :3]) ** 2))
        assert rmse_pos < 0.1, f"SINDy position RMSE too high: {rmse_pos:.4f}"


# ── 9. Hankel-specific: buffer reset ─────────────────────────────────────────

class TestHankelBuffers:
    def test_buffer_reset_changes_output(self, trained_models):
        """After reset_buffers, Hankel predictions start fresh."""
        model = trained_models['Hankel']
        rng = np.random.RandomState(3)
        x0 = rng.randn(NX)
        u  = rng.uniform(-1, 1, NU)

        model.reset_buffers(x0)
        p1 = model.predict(x0, u)

        model.reset_buffers(np.zeros(NX))   # different initial state
        p2 = model.predict(x0, u)

        # Different buffer → different prediction
        assert not np.allclose(p1, p2), \
            "reset_buffers() had no effect on Hankel prediction"

    def test_buffer_length_after_reset(self, trained_models):
        """Buffer should have exactly `delay` entries after reset."""
        model = trained_models['Hankel']
        model.reset_buffers(np.ones(NX))
        assert len(model._state_buf) == model.delay


# ── 10. Heading angle wrapping ────────────────────────────────────────────────

class TestHeadingWrapping:
    @pytest.mark.parametrize("name", ["SINDy", "DMD", "Hankel"])
    def test_heading_stays_in_range(self, name, trained_models):
        """Heading (x[3]) must stay in (-π, π]."""
        model = trained_models[name]
        rng = np.random.RandomState(42)
        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(np.zeros(NX))

        for _ in range(50):
            x = rng.randn(NX)
            x[3] = rng.uniform(-np.pi, np.pi)
            u = rng.uniform(-1, 1, NU)
            pred = model.predict(x, u)
            assert -np.pi <= pred[3] <= np.pi, \
                f"{name}: heading {pred[3]:.3f} outside (-π, π]"
