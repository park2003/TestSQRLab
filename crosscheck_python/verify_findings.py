import sys; sys.path.insert(0, '/home/claude/testsqrlab/crosscheck_python')
import numpy as np
from scipy.special import lambertw, eval_laguerre
from scipy.optimize import brentq
from engine import qcrb_theory
f = lambda y: (2 / y) * (1 - np.exp(-y / 2))

print("[1] y*: brentq vs Lambert W_{-1}")
ys = brentq(lambda y: (1 + y) * np.exp(-y / 2) - 1, 1, 5)
yw = (-2 * lambertw(-1 / (2 * np.sqrt(np.e)), -1) - 1).real
print(f"    brentq {ys:.12f}  Lambert {yw:.12f}  const 4y/(1+y)^2 {4*ys/(1+ys)**2:.6f}  (1+y)^2/4y {(1+ys)**2/(4*ys):.6f}")

print("[2] integer m*(Y): formula vs brute force over m = 1..400, Y in (0, 300]")
g = lambda m, Y: m * (1 - np.exp(-Y / (2 * m))) ** 2
bad = 0
for Y in np.linspace(0.01, 300, 30000):
    brute = 1 + int(np.argmax([g(m, Y) for m in range(1, 401)]))
    if Y <= 4 * np.log(1 + np.sqrt(2)): form = 1
    else:
        c = {max(1, int(np.floor(Y / ys))), int(np.ceil(Y / ys))}; form = max(c, key=lambda m: g(m, Y))
    bad += brute != form
print(f"    mismatches: {bad} / 30000")

print("[3] fixed per-round x: instant cutoff and Newton m_x vs brute force (first integer m>1 after peak with R<1)")
R = lambda m, x: (1 - np.exp(-m * x / 2)) ** 2 / (m * (1 - np.exp(-x / 2)) ** 2)
print(f"    2 ln(1+sqrt2) = {2*np.log(1+np.sqrt(2)):.6f}; R(2, 1.76) = {R(2,1.76):.6f}, R(2, 1.77) = {R(2,1.77):.6f}")
for x in (0.01, 0.05, 0.2, 0.5, 1.0, 1.5, 1.75):
    c = 1 - np.exp(-x / 2); m = 1 / c ** 2; steps = []
    for _ in range(6):
        gm = 1 - np.exp(-m * x / 2) - c * np.sqrt(m); dg = (x / 2) * np.exp(-m * x / 2) - c / (2 * np.sqrt(m))
        m -= gm / dg; steps.append(m)
    mm = np.arange(2, 200000); first = mm[np.argmax(R(mm, x) < 1)]
    print(f"    x={x:<5} seed {1/c**2:10.2f}  Newton {steps[1]:10.4f} {steps[2]:10.4f} {steps[3]:10.4f} -> ceil {int(np.ceil(steps[-1]))}   brute {first}")

print("[4] optimal roots y_{n,0}: largest root of L_n - y L_n + 2y L_n' = 0 (manuscript eq. 79)")
def F(y, n):
    Ln = eval_laguerre(n, y); dL = 0.0 if n == 0 else -eval_laguerre(n - 1, y) * 0  # placeholder
    return Ln
from numpy.polynomial import laguerre as L
yn = {}
for n in (0, 1, 2, 3, 5):
    c = np.zeros(n + 1); c[n] = 1
    p = L.lag2poly(c); P = np.polynomial.Polynomial(p)
    eq = P - np.polynomial.Polynomial([0, 1]) * P + 2 * np.polynomial.Polynomial([0, 1]) * P.deriv()
    r = eq.roots(); r = r[np.isreal(r)].real; r = r[r > 0]
    # global optimum: maximize y e^{-y} L_n(y)^2 among roots
    best = max(r, key=lambda y: y * np.exp(-y) * P(y) ** 2); yn[n] = best
    V = np.exp(-best / 2) * P(best); Cn = 4 * best / (1 - V ** 2)
    print(f"    n={n}: roots {np.round(np.sort(r),3)}  y_n0 = {best:.4f}  V = {V:+.4f}  C_n = {Cn:.2f}  M(1%) = {np.sqrt(0.01/Cn):.4f}  M(5%) = {np.sqrt(0.05/Cn):.4f}  first slope y-0.5 = {best-0.5:.3f}")

print("[5] cooling (r=0, nbar=0): exponential-trajectory thresholds vs exact closed form")
al = np.arange(0.005, 4.5, 0.0005)
Qc = lambda n, x: min(qcrb_theory(a, n, 0.0, x, 0.0) for a in al[::4])
for n in (1, 3, 5):
    lam = yn[n] - 0.5
    exact = brentq(lambda Y: np.log(2 * Qc(n, Y / 2) / Qc(n, Y)), 0.02, 1.5, xtol=1e-4)
    print(f"    n={n}: lambda={lam:.3f}  2ln2/lambda = {2*np.log(2)/lam:.4f}   exact closed-form 1-vs-2 = {exact:.4f}")
print("    fixed per-round x: Lambert-W m_x vs exact closed form (first m>1 with R<1)")
for n in (1, 3):
    lam = yn[n] - 0.5
    for x in (0.005, 0.02, 0.05, 0.1):
        z = lam * x
        mw = 2 if z >= 1 else int(np.ceil((-lambertw(-z * np.exp(-z), -1) / z).real))
        Q1 = Qc(n, x); ex = None
        for m in range(2, 400):
            if m * Q1 / Qc(n, m * x) < 1: ex = m; break
        print(f"      n={n} x={x:<6} lambda x={z:.3f}: Lambert {mw:>4}   exact {ex}")
