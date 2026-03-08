# Data-Driven Approaches for Underwater Vehicle Control — Comparative Survey

A comprehensive benchmark comparing **seven data-driven dynamics identification methods**
applied to the [BlueROV2 Heavy](https://bluerobotics.com/store/rov/bluerov2-heavy/) 4-DOF
autonomous underwater vehicle, with evaluation on one-step prediction, multi-step rollout,
disturbance robustness, and MPC tracking performance.

---

## Methods Compared

| Method | Type | Key Property |
|--------|------|--------------|
| **Classic** | Analytical | Ground-truth RK4 baseline |
| **NN** | Black-box | Flexible, requires large data |
| **PINN** | Physics-residual | Analytically grounded, data-efficient |
| **SINDy** | Sparse regression | Interpretable, physically consistent |
| **DMD** | Linear data-driven | Fastest, admits linear MPC |
| **GP** | Probabilistic | Uncertainty quantification |
| **Hankel-DMD** | Delay-embedded linear | Captures slower modes |

---

## Repository Structure

```
survey/
├── src/
│   ├── bluerov_sim.py      # BlueROV2 simulator (RK4, lemniscate trajectory)
│   └── parameters.py       # Physical parameters (mass, damping, limits)
├── methods/
│   ├── base.py             # Abstract BaseModel (train/predict/rollout)
│   ├── classic_model.py    # Analytical baseline
│   ├── nn_model.py         # Standard MLP (PyTorch)
│   ├── pinn_model.py       # Physics-Informed NN (residual learning)
│   ├── sindy_model.py      # SINDy (sequential thresholded least squares)
│   ├── dmd_model.py        # Dynamic Mode Decomposition
│   ├── gp_model.py         # Gaussian Process Regression (sklearn)
│   └── hankel_model.py     # Hankel-DMD (time-delay embedding)
├── mpc/
│   ├── mpc_controller.py   # Random-shooting MPC (model-agnostic)
│   └── inverse_controller.py # Analytical inverse-dynamics controller
├── experiments/
│   └── run_survey.py       # Main experiment runner
├── utils/
│   └── plot_survey.py      # Spider diagrams, bar charts, tables
├── results/
│   ├── summary.csv         # Combined results table
│   ├── onestep_results.csv
│   ├── rollout_results.csv
│   └── figures/            # PDF figures + LaTeX tables
└── paper/
    ├── main.tex            # Full IEEE-style survey paper
    └── references.bib
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy scipy scikit-learn torch pandas matplotlib
```

### 2. Run experiments

```bash
# From the repo root
python -m survey.experiments.run_survey

# Skip MPC (faster, ~3 min total)
python -m survey.experiments.run_survey --skip-mpc
```

### 3. Generate figures only

```bash
python -m survey.utils.plot_survey
```

### 4. Compile paper

```bash
cd survey/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Key Results

| Method | Pos. RMSE (clean) | Rollout RMSE | Train Time | Pred. Time |
|--------|-------------------|--------------|------------|------------|
| Classic | 0.00000 | 0.0000 | <1 ms | 0.03 ms |
| **PINN** | **~0.00000** | **~0.000** | 32 s | 0.21 ms |
| **SINDy** | **0.01527** | **0.1421** | 0.47 s | 0.13 ms |
| DMD | 0.01602 | 0.6627 | 3.4 ms | **0.004 ms** |
| GP | 0.01634 | 0.3092 | 41.7 s | 0.88 ms |
| NN | 0.14466 | 3.1153 | 41.7 s | 0.16 ms |
| Hankel | 0.14483 | 0.7016 | 0.11 s | 0.008 ms |

**Key takeaways:**
- **PINN** (residual on analytical model) achieves near-perfect accuracy when a partial model is available.
- **SINDy** is the best purely data-driven method: accurate, stable rollouts, fast, interpretable.
- **DMD** is fastest by 30×, ideal for high-rate embedded MPC.
- **GP** uniquely provides uncertainty estimates for safety-critical applications.
- **NN** needs larger datasets and regularisation to avoid rollout divergence.

---

## Vehicle Model

The BlueROV2 Heavy simulator implements the SNAME 4-DOF equations of motion:

```
(m - X_ud) * u_dot = X + (m - Y_vd)*v*r + (X_u + X_uc*|u|)*u
(m - Y_vd) * v_dot = Y - (m - X_ud)*u*r + (Y_v + Y_vc*|v|)*v
(m - Z_wd) * w_dot = Z + (Z_w + Z_wc*|w|)*w - m*g + F_bouy
(I_zz - N_rd) * r_dot = Mz + (X_ud - Y_vd)*u*v + (N_r + N_rc*|r|)*r
```

Parameters: `m=11.4 kg`, `MAX_FORCE=40 N`, `MAX_TORQUE=10 Nm`, `dt=0.05 s`.

---

## MPC Architecture

The `MPCController` uses **random shooting** with `N=150` candidate action sequences
over `H=8` steps, evaluated against a position + heading cost. This is model-agnostic
and works with any `BaseModel`.

For real-time applications, the linear structure of DMD and SINDy enables standard
**linear MPC** formulations with QP solvers, reducing computation by orders of magnitude.

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{survey_auv_dd_2024,
  title  = {Data-Driven Approaches for Underwater Vehicle Control: A Comparative Survey},
  author = {Anonymous},
  year   = {2024},
  url    = {https://github.com/abdelhakim96/ContinualLearning_Drone}
}
```
