import sys; sys.path.insert(0, '/home/claude/testsqrlab/crosscheck_python')
import numpy as np
from scipy.optimize import brentq
from engine import qcrb_theory
al = np.linspace(0.01, 6, 6000)
Qc = lambda n, Y: np.min(qcrb_theory(al, n, 0.0, Y, 0.0))
for n in (1, 3, 5):
    Ys = np.concatenate([np.linspace(0.005, 1, 800), np.linspace(1, 25, 600)])
    d = [Qc(n, Y) - 2 * Qc(n, Y / 2) for Y in Ys]
    cr = [brentq(lambda Y: Qc(n, Y) - 2 * Qc(n, Y / 2), Ys[i], Ys[i+1]) for i in range(len(Ys)-1) if np.sign(d[i]) != np.sign(d[i+1])]
    print(n, np.round(cr, 4), "sign at Y=25:", "A(2 rounds) better" if d[-1] > 0 else "one exposure better")
