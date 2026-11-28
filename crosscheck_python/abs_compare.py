import sys; sys.path.insert(0, '.')
import numpy as np
from engine import propagate, fock, displace_matrix, qfi_from_C
D = 61; NS = range(6)
alg = np.round(np.arange(0.01, 4.5, 0.01), 4)
V = [displace_matrix(2j * a, D) for a in alg]
f = lambda y: (2 / y) * (1 - np.exp(-y / 2))
Q0 = {0: 0.679570, 1: 0.374884, 2: 0.300640, 3: 0.256426, 5: 0.213455}

# [1] additivity: QFI of rho_spin^{(x)m} computed directly (SLD eigen-formula) vs m * F_1
def qfi_sld(r, dr):
    w, U = np.linalg.eigh(r); dd = U.conj().T @ dr @ U; F = 0.0
    for i in range(len(w)):
        for j in range(len(w)):
            if w[i] + w[j] > 1e-12: F += 2 * abs(dd[i, j]) ** 2 / (w[i] + w[j])
    return F
rho, drho = propagate(fock(3, D), 0.1, 2.5, 0, 0, d=D)
q = [1 / qfi_from_C(np.trace(v @ rho), np.trace(v @ drho)) for v in V]; k = int(np.argmin(q)); v = V[k]
C, dC = np.trace(v @ rho), np.trace(v @ drho)
rs = 0.5 * np.array([[1, np.conj(C)], [C, 1]]); drs = 0.5 * np.array([[0, np.conj(dC)], [dC, 0]])
F1 = qfi_sld(rs, drs)
print(f"[1] one round (n=3, beta=0.1, Gphi t=2.5, alpha*={alg[k]}): F1 = {F1:.6f} (formula {1/q[k]:.6f})")
R, dR = rs, drs
for m in range(2, 6):
    dR = np.kron(dR, rs) + np.kron(R, drs); R = np.kron(R, rs)      # d/dbeta of the m-fold product (Leibniz)
    print(f"    m={m}: QFI of the {2**m}x{2**m} product state = {qfi_sld(R, dR):.6f}   m*F1 = {m*F1:.6f}")

# [2] absolute comparison at fixed budget Gphi T = 20: Var(M) = m * QCRB_round, vs the summation formula m*QCRB0/f(20/m)^2
cache = {}
def qr(n, b, g):
    key = (n, round(b, 9), round(g, 9))
    if key not in cache:
        r_, d_ = propagate(fock(n, D), b, g, 0, 0, d=D)
        cache[key] = min(1 / qfi_from_C(np.trace(v @ r_), np.trace(v @ d_)) for v in V)
    return cache[key]
GT = 20.0; ms = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40]
print(f"\n[2] fixed budget Gphi*T = {GT}; entries are Var(M) = m * QCRB_round (absolute); 'sum' = m*QCRB0(n)/f(20/m)^2 (exact small-signal)")
for M in (0.1, 0.3, 0.5, 0.8):
    print(f"  M = {M}")
    print("     m |  n=0     n=1     n=2     n=3     n=5   | best (n)      | sum n=5 | best / sum_n5")
    for m in ms:
        vals = {n: m * qr(n, M / m, GT / m) for n in (0, 1, 2, 3, 5)}
        nb = min(vals, key=vals.get); s5 = m * Q0[5] / f(GT / m) ** 2
        print(f"    {m:>2} | " + " ".join(f"{vals[n]:7.3f}" for n in (0, 1, 2, 3, 5)) + f" | {vals[nb]:7.3f} ({nb}) | {s5:7.3f} | {vals[nb]/s5:6.3f}")
    sys.stdout.flush()
