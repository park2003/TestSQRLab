import sys; sys.path.insert(0, '/home/claude/testsqrlab/crosscheck_python')
import numpy as np
from scipy.optimize import minimize_scalar
from engine import qcrb_theory
yn = {0: 1.0, 1: 3.7320508, 3: 10.5731}
def Qopt(n, r, nb, e):
    a0 = np.sqrt(yn[n]) / 2 * np.exp(r)
    res = minimize_scalar(lambda a: np.log(qcrb_theory(a, n, r, e*(nb+1), e*nb)), bounds=(0.8*a0, 1.2*a0), method='bounded', options={'xatol':1e-12})
    return res.fun
for n in (0, 1, 3):
    for r, nb in ((0, 0), (0, 1), (0.5, 0), (0.5, 2)):
        h = 1e-5; s = (Qopt(n, r, nb, h) - Qopt(n, r, nb, 0)) / h
        y = yn[n]; mine = y*np.exp(2*r)*(2*nb+1) - 0.5; ms = np.exp(2*r)*(2*nb+1)*(y-1) + 0.5
        print(f"n={n} r={r} nbar={nb}: numeric {s:9.4f}  y e^2r(2nb+1)-1/2 = {mine:9.4f}  eq114 = {ms:9.4f}")
