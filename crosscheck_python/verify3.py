import sys; sys.path.insert(0, '/home/claude/testsqrlab/crosscheck_python')
import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq
from scipy.linalg import expm
from engine import qcrb_theory, propagate, qcrb_ideal_curve, fock, DIM
print("DIM", DIM)
al = np.linspace(0.01, 4.5, 4000); alE = np.linspace(0.02, 3.0, 600)
yn = {1: 3.7320508, 3: 10.5731}
Qc = lambda n, Y: np.min(qcrb_theory(al, n, 0.0, Y, 0.0))
print("[5b'] Lambert vs exact over a wider x range")
for n in (1, 3):
    lam = yn[n] - 0.5
    for x in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        z = lam * x; mw = 2 if z >= 1 else int(np.ceil((-lambertw(-z*np.exp(-z), -1)/z).real))
        Q1 = Qc(n, x); ex = next((m for m in range(2, 5000) if m * Q1 < Qc(n, m * x)), None)
        print(f"   n={n} x={x}: z={z:.3f} Lambert {mw}  exact {ex}")
f = lambda y: (2/y)*(1-np.exp(-y/2))
rho, drho = propagate(fock(5), 1e-4, 0, 0, 0); q5 = qcrb_ideal_curve(rho, drho, alE).min()
print(f"[8'] engine QCRB0(n=5) full grid = {q5:.5f}; Var_opt(Y=20) = {1.2277*20*q5:.3f}; m=8 -> {q5*400/(32*(1-np.exp(-1.25))**2):.3f}")
d = DIM; a = np.diag(np.sqrt(np.arange(1, d)), 1)
for n in (1, 3):
    r = np.arcsinh(np.sqrt(n)); cands = [expm(s*0.5*r*(a@a - a.T@a.T)) @ fock(0, d) for s in (1, -1)]
    q = [qcrb_ideal_curve(*propagate(p, 1e-4, 0, 0, 0), alE).min() for p in cands]; psi = cands[int(np.argmin(q))]
    qf0 = qcrb_ideal_curve(*propagate(fock(n), 1e-4, 0, 0, 0), alE).min()
    g = lambda X: np.log(qcrb_ideal_curve(*propagate(psi, 1e-4, X, 0, 0), alE).min() / (qf0 / f(X)**2))
    print(f"[9'] n={n}: sq0 {min(q):.5f} (theory {np.exp(1)/4*np.exp(-2*r):.5f}), fock0 {qf0:.5f}; crossover X = {brentq(g, 0.005, 0.4, xtol=1e-4):.4f}")
