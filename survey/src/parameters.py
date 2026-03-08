"""BlueROV2 4-DOF parameters (shared across all methods)."""

# Vehicle
m = 11.4;  g = 9.82
F_bouy = 1026 * 0.0115 * g

# Added mass
X_ud = -2.6;  Y_vd = -18.5;  Z_wd = -13.3;  N_rd = -0.28

# Inertia
I_zz = 0.245

# Linear damping
X_u = -0.09;  Y_v = -0.26;  Z_w = -0.19;  N_r = -4.64

# Quadratic damping
X_uc = -34.96;  Y_vc = -103.25;  Z_wc = -74.23;  N_rc = -0.43

# Limits
MAX_FORCE = 40.0;  MAX_TORQUE = 10.0
