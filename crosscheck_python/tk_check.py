import sys
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from engine import DIM, ladder, displace_matrix, propagate, NoiseStep

a = ladder(DIM); at = a.T; nop = at @ a; G = at - a

def lindblad_rhs(beta, Gphi, Gc, Gh):
    def f(t, y):
        r = y.reshape(DIM, DIM)
        d = beta * (G @ r - r @ G)                       # -i[H, rho] with H = i beta (a^dag - a)
        for L, g in ((nop, Gphi), (a, Gc), (at, Gh)):
            if g: d += g * (L @ r @ L.T - 0.5 * (L.T @ L @ r + r @ L.T @ L))
        return d.ravel()
    return f

def rho_master(psi, beta, Gphi=0., Gc=0., Gh=0., T=1.0):
    r0 = np.outer(psi, psi).ravel()
    s = solve_ivp(lindblad_rhs(beta, Gphi, Gc, Gh), (0, T), r0, method='DOP853', rtol=1e-12, atol=1e-14)
    return s.y[:, -1].reshape(DIM, DIM)

def lie_kraus(psi, beta, Gphi, Gc, Gh, S):            # the 1104 scheme
    D = expm((beta / S) * G); r = np.outer(psi, psi)
    idx = np.arange(DIM); mask = np.exp(-0.5 * Gphi / S * (idx[:, None] - idx[None, :])**2)
    M0c = np.eye(DIM) - 0.5 * Gc / S * nop; M1c = np.sqrt(Gc / S) * a
    M0h = np.eye(DIM) - 0.5 * Gh / S * (nop + np.eye(DIM)); M1h = np.sqrt(Gh / S) * at
    for _ in range(S):
        r = D @ r @ D.T
        r = r * mask
        if Gc: r = M0c @ r @ M0c.T + M1c @ r @ M1c.T
        if Gh: r = M0h @ r @ M0h.T + M1h @ r @ M1h.T
    return r

tdist = lambda A, B: 0.5 * np.abs(np.linalg.eigvalsh(A - B)).sum()

psi = displace_matrix(1.5)[:, 1]
rb = NoiseStep(1.0, 0.3, 0.1, 0.37).apply(np.outer(psi, psi))
rr = rho_master(psi, 0.0, 1.0, 0.3, 0.1, T=0.37)
print(f"[1] noise step alone: trace distance {tdist(rb, rr):.2e}, max |diff| {np.abs(rb-rr).max():.2e}, trace {np.trace(rb):.15f}")

psi0 = np.zeros(DIM); psi0[1] = 1.0
for (Gp, Gc, Gh) in ((1.0, 0.3, 0.1), (3.0, 0.0, 0.0)):
    ref = rho_master(psi0, 2.0, Gp, Gc, Gh)
    print(f"[2] beta=2 Gphi={Gp} Gc={Gc} Gh={Gh}")
    for S in (16, 32, 64, 128, 256):
        rs, _ = propagate(psi0, 2.0, Gp, Gc, Gh, steps=S)
        rl = lie_kraus(psi0, 2.0, Gp, Gc, Gh, S)
        print(f"    S={S:4d}  Strang+T(k): {tdist(rs, ref):.3e}   Lie+Kraus (1104): {tdist(rl, ref):.3e}  trace err {abs(np.trace(rl)-1):.1e}")
