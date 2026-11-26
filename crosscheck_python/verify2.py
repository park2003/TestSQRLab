import sys; sys.path.insert(0, '/home/claude/testsqrlab/crosscheck_python')
import numpy as np
from scipy.special import lambertw, eval_laguerre as Lag
from scipy.optimize import brentq, minimize_scalar
from scipy.linalg import expm
from engine import qcrb_theory, propagate, qcrb_ideal_curve, fock, DIM
yn = {0: 1.0, 1: 3.7320508, 2: 7.0561, 3: 10.5731, 5: 17.8626}
al = np.linspace(0.01, 4.5, 4000)
def Qc(n, Y): return np.min(qcrb_theory(al, n, 0.0, Y, 0.0))
print("[5a] cooling-only fixed-budget 1-vs-2: Var1 = Q(Y), Var2 = 2 Q(Y/2)")
for n in (0, 1, 3, 5):
    Ys = np.linspace(0.005, 5, 1000); d = [Qc(n, Y) - 2 * Qc(n, Y / 2) for Y in Ys]
    cr = [Ys[i] for i in range(len(Ys) - 1) if np.sign(d[i]) != np.sign(d[i + 1])]
    ex = [brentq(lambda Y: Qc(n, Y) - 2 * Qc(n, Y / 2), c - 0.006, c + 0.006) for c in cr]
    lam = yn[n] - 0.5
    print(f"   n={n}: exact sign changes at Y = {np.round(ex,4)}   2ln2/lambda = {2*np.log(2)/lam:.4f}")
print("[5b] fixed per-round x: Lambert-W m_x vs exact closed form, cooling only")
for n in (1, 3):
    lam = yn[n] - 0.5
    for x in (0.005, 0.01, 0.02, 0.05):
        z = lam * x
        mw = 2 if z >= 1 else int(np.ceil((-lambertw(-z * np.exp(-z), -1) / z).real))
        Q1 = Qc(n, x); ex = next((m for m in range(2, 2000) if m * Q1 < Qc(n, m * x)), None)
        print(f"   n={n} x={x:<6} z={z:.3f}: Lambert {mw:>4}   exact {ex}")
print("[6] manuscript eq. 103 crossovers (r=0, cold), t_sig = 0.00698 s")
Q0 = {n: np.exp(yn[n]) / (4 * yn[n] * Lag(n, yn[n]) ** 2) for n in yn}
for n in (0,1,2,3,5): print(f"   QCRB_{n}(0) = {Q0[n]:.5f}")
for A, B in ((1, 0), (2, 1), (3, 2)):
    e = np.log(Q0[B] / Q0[A]) / (yn[A] - yn[B])
    print(f"   n={A} crosses n={B}: Gamma t = {e:.4f}  Gamma = {e/0.00698:.2f} /s   (eq.100-103 with y2=7.23 would give", end=" ")
    y2 = {**yn, 2: 7.23}; print(f"{np.log(Q0[B]/Q0[A])/(y2[A]-y2[B]):.4f})")
print("[7] exact (closed form, fixed b, varying Gamma t) n vs n-1 crossings for comparison")
for A, B in ((1, 0), (2, 1), (3, 2)):
    g = lambda e: np.log(Qc(A, e) / Qc(B, e))
    es = np.linspace(0.001, 1.0, 400); v = [g(e) for e in es]
    i = next(i for i in range(len(es) - 1) if v[i] < 0 <= v[i + 1])
    print(f"   n={A} vs n={B}: exact Gamma t = {brentq(g, es[i], es[i+1]):.4f}")
print("[8] beta->0 dephasing, n=5, Gphi T = 20: Var(M) one shot vs optimal division (closed form) and engine check")
Q05 = None
rho, drho = propagate(fock(5), 1e-4, 0.0, 0.0, 0.0); Q05 = qcrb_ideal_curve(rho, drho, al[:1500]).min()
f = lambda y: (2/y)*(1-np.exp(-y/2))
V1 = Q05 / f(20) ** 2; Vm = min(Q05 * 20**2 / (4*m*(1-np.exp(-20/(2*m)))**2) for m in range(1, 60))
print(f"   QCRB0(n=5) = {Q05:.5f}; Var1 = {V1:.4f}, Var_opt = {Vm:.4f}, ratio {V1/Vm:.3f}, 1.2277*Y*Q0 = {1.2277*20*Q05:.4f}")
rho, drho = propagate(fock(5), 1e-4, 2.5, 0.0, 0.0)
print(f"   engine n=5 Gphi t=2.5: {qcrb_ideal_curve(rho, drho, al[:1500]).min():.5f} vs Q0/f^2 {Q05/f(2.5)**2:.5f}")
print("[9] beta->0: squeezed vacuum (sinh^2 r = n) vs Fock n under dephasing X = Gphi t")
d = DIM; a = np.diag(np.sqrt(np.arange(1, d)), 1)
def sq(r):  # squeeze along the readout quadrature (engine readout D(2i alpha) -> squeeze p)
    out = []
    for s in (+1, -1):
        S = expm(s * 0.5 * r * (a @ a - a.T @ a.T)); out.append(S @ fock(0, d))
    return out
for n in (1, 3):
    r = np.arcsinh(np.sqrt(n)); cands = sq(r)
    q0s = []
    for psi in cands:
        rho, drho = propagate(psi, 1e-4, 0.0, 0.0, 0.0); q0s.append(qcrb_ideal_curve(rho, drho, al[:1500]).min())
    psi = cands[int(np.argmin(q0s))]
    rho, drho = propagate(fock(n), 1e-4, 0.0, 0, 0); qf0 = qcrb_ideal_curve(rho, drho, al[:1500]).min()
    line = f"   n={n} (r={r:.3f}) X=0: sq {min(q0s):.4f} fock {qf0:.4f} |"
    for X in (0.02, 0.05, 0.1, 0.2, 0.5):
        rs, ds = propagate(psi, 1e-4, X, 0, 0); qs = qcrb_ideal_curve(rs, ds, al[:1500]).min()
        line += f" X={X}: sq {qs:.4f} fock {qf0/f(X)**2:.4f} |"
    print(line)
