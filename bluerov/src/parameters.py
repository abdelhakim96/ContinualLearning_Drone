"""
BlueROV2 Heavy Physical Parameters
Reference: Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion Control.
"""

# ─── Vehicle Properties ────────────────────────────────────────────────────────
m = 11.4          # Mass (kg)
g = 9.82          # Gravitational acceleration (m/s^2)
F_bouy = 1026 * 0.0115 * g  # Buoyancy force (N)  [seawater density * volume * g]

# ─── Added Mass Coefficients (kg / kg·m^2) ────────────────────────────────────
X_ud = -2.6       # Surge added mass
Y_vd = -18.5      # Sway added mass
Z_wd = -13.3      # Heave added mass
K_pd = -0.054     # Roll added inertia
M_qd = -0.0173    # Pitch added inertia
N_rd = -0.28      # Yaw added inertia

# ─── Moments of Inertia (kg·m^2) ──────────────────────────────────────────────
I_xx = 0.21
I_yy = 0.245
I_zz = 0.245

# ─── Linear Damping Coefficients ──────────────────────────────────────────────
X_u  = -0.09      # Surge linear damping  (N·s/m)
Y_v  = -0.26      # Sway linear damping   (N·s/m)
Z_w  = -0.19      # Heave linear damping  (N·s/m)
K_p  = -0.895     # Roll linear damping   (N·s/rad)
M_q  = -0.287     # Pitch linear damping  (N·s/rad)
N_r  = -4.64      # Yaw linear damping    (N·s/rad)

# ─── Quadratic (Nonlinear) Damping Coefficients ───────────────────────────────
X_uc = -34.96     # Surge quadratic damping  (N·s^2/m^2)
Y_vc = -103.25    # Sway quadratic damping   (N·s^2/m^2)
Z_wc = -74.23     # Heave quadratic damping  (N·s^2/m^2)
K_pc = -0.084     # Roll quadratic damping   (N·s^2/rad^2)
M_qc = -0.028     # Pitch quadratic damping  (N·s^2/rad^2)
N_rc = -0.43      # Yaw quadratic damping    (N·s^2/rad^2)

# ─── Centre of Buoyancy (relative to CG) ─────────────────────────────────────
z_b  = -0.1       # CB below CG along z-axis (m)

# ─── Control Input Limits ─────────────────────────────────────────────────────
MAX_FORCE  = 40.0   # Maximum thruster force per axis (N)
MAX_TORQUE = 10.0   # Maximum torque per axis (N·m)
