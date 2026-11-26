import sys; sys.path.insert(0, '/home/claude/testsqrlab/crosscheck_python')
import numpy as np
from engine import qcrb_theory
al = np.linspace(0.01, 4.5, 4000)
Qc = lambda n, Y: np.min(qcrb_theory(al, n, 0.0, Y, 0.0))
for n in (0, 1, 3):
    for x in (0.01, 0.05, 0.1, 0.5):
        Q1 = Qc(n, x); pred = 16 * Q1 / (np.e * x**2)
        ex = next((m for m in range(2, 200000) if m * Q1 < Qc(n, m * x)), None) if x >= 0.05 or n else None
        print(f"n={n} x={x}: Q(x)={Q1:.4f} 16Q/(e x^2)={pred:8.1f} exact {ex}")
# large-Y asymptote check
for n in (0, 1, 3):
    print(n, [round(Qc(n, Y) / (np.e * Y**2 / 16), 4) for Y in (5, 10, 20, 50)])
