"""
BlueROV2 4-DOF Simulator.

State  x = [x, y, z, psi, u, v, w, r]   (8-D, angle in rad)
Input  u = [X, Y, Z, Mz]                 (4-D body-frame forces)
"""
import numpy as np
from .parameters import (m, g, F_bouy, X_ud, Y_vd, Z_wd, N_rd, I_zz,
                          X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc, N_r, N_rc,
                          MAX_FORCE, MAX_TORQUE)

NX = 8  # state dim
NU = 4  # control dim

def ssa(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def dynamics(x, u, dist=None, unc=0.0):
    """Continuous-time f(x,u) → dx/dt.  unc scales params by (1+unc)."""
    sc = 1.0 + unc
    m_ = m*sc;  Xu_ = X_u*sc;  Xuc_ = X_uc*sc
    Yv_ = Y_v*sc;  Yvc_ = Y_vc*sc;  Zw_ = Z_w*sc;  Zwc_ = Z_wc*sc
    Nr_ = N_r*sc;  Nrc_ = N_rc*sc

    psi = x[3];  u_b, v_b, w_b, r_b = x[4], x[5], x[6], x[7]
    Fx, Fy, Fz, Mz = u[0], u[1], u[2], u[3]
    if dist is not None:
        Fx += dist[0]; Fy += dist[1]; Fz += dist[2]; Mz += dist[3]

    Mu = m_ - X_ud;  Mv = m_ - Y_vd;  Mw = m_ - Z_wd;  Jr = I_zz - N_rd

    dx = np.zeros(8)
    dx[0] = np.cos(psi)*u_b - np.sin(psi)*v_b
    dx[1] = np.sin(psi)*u_b + np.cos(psi)*v_b
    dx[2] = w_b
    dx[3] = r_b
    dx[4] = (Fx + (m_-Y_vd)*v_b*r_b + (Xu_+Xuc_*abs(u_b))*u_b) / Mu
    dx[5] = (Fy - (m_-X_ud)*u_b*r_b + (Yv_+Yvc_*abs(v_b))*v_b) / Mv
    dx[6] = (Fz + (Zw_+Zwc_*abs(w_b))*w_b + m_*g - F_bouy) / Mw
    dx[7] = (Mz - (X_ud-Y_vd)*u_b*v_b + (Nr_+Nrc_*abs(r_b))*r_b) / Jr
    return dx

def rk4(x, u, dt, dist=None, unc=0.0):
    k1 = dynamics(x, u, dist, unc)
    k2 = dynamics(x+dt/2*k1, u, dist, unc)
    k3 = dynamics(x+dt/2*k2, u, dist, unc)
    k4 = dynamics(x+dt*k3, u, dist, unc)
    xn = x + dt/6*(k1+2*k2+2*k3+k4)
    xn[3] = ssa(xn[3])
    return xn

def collect_data(N=5000, dt=0.05, seed=42):
    """Generate (x, u, x_next) training data with random excitation."""
    rng = np.random.RandomState(seed)
    X, U, Xn = [], [], []
    x = np.zeros(8)
    for _ in range(N):
        u_cmd = rng.uniform([-MAX_FORCE,-MAX_FORCE,-MAX_FORCE,-MAX_TORQUE],
                            [ MAX_FORCE, MAX_FORCE, MAX_FORCE, MAX_TORQUE])
        xn = rk4(x, u_cmd, dt)
        X.append(x.copy()); U.append(u_cmd.copy()); Xn.append(xn.copy())
        x = xn
        if rng.rand() < 0.05:  # occasional reset
            x = rng.randn(8)*np.array([3,3,2,np.pi,1,1,0.5,0.5])
    return np.array(X), np.array(U), np.array(Xn)

def lemniscate_ref(t, a=3.0, b=1.5, z=-2.0, om=0.25):
    """Reference trajectory → (N,4) [x,y,z,psi]."""
    x = a*np.sin(om*t)
    y = b*np.sin(om*t)*np.cos(om*t)
    z_ = z*np.ones_like(t)
    dx = a*om*np.cos(om*t)
    dy = b*om*(np.cos(om*t)**2 - np.sin(om*t)**2)
    psi = np.arctan2(dy, dx)
    return np.column_stack([x, y, z_, psi])
