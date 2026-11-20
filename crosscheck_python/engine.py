"""Python port of the 1113 QCRB engine (Strang split, exact noise step, exact d/dbeta).

Dimensionless units: exposure t_sig = 1, signal drive rate beta, noise strengths Gamma*t_sig.
Readout: C = Tr[D(2i alpha) rho], qubit QFI from the Bloch vector.
"""
import numpy as np
from scipy.linalg import expm
from numpy.polynomial.laguerre import lagval

DIM = 51


def ladder(d):
    return np.diag(np.sqrt(np.arange(1, d)), 1)          # annihilation a


def displace_matrix(xi, d=DIM, pad=None):
    pad = pad or 2 * d
    a = ladder(pad)
    return expm(xi * a.T - np.conj(xi) * a)[:d, :d]


def fock(n, d=DIM):
    v = np.zeros(d); v[n] = 1.0
    return v


# ---------------------------------------------------------------- exact noise channel
class NoiseStep:
    """exp(L_noise * frac), one tridiagonal block T^(k) per superdiagonal k."""

    def __init__(self, Gphi, Gc, Gh, frac, d=DIM):
        N = d - 1
        self.d = d
        idx = np.arange(d)
        self.mask = np.exp(-0.5 * Gphi * frac * (idx[:, None] - idx[None, :]) ** 2)
        self.diagonal_only = (Gc == 0 and Gh == 0)
        if self.diagonal_only:
            return
        up = lambda j: j + 1.0 if j < N else 0.0          # <j|a a^dag|j>, cut at N -> trace conserved
        self.E = []
        for k in range(d):
            L = N - k + 1
            T = np.zeros((L, L))
            for i in range(L):
                m = i
                T[i, i] = -0.5 * Gc * (2 * m + k) - 0.5 * Gh * (up(m) + up(m + k)) - 0.5 * Gphi * k ** 2
                if i < L - 1:
                    T[i, i + 1] = Gc * np.sqrt((m + 1) * (m + k + 1))
                if i > 0:
                    T[i, i - 1] = Gh * np.sqrt(m * (m + k))
            self.E.append(expm(T * frac))

    def apply(self, rho):
        """Works for any (also non-Hermitian, complex) operator: super- and subdiagonals use the same block."""
        if self.diagonal_only:
            return rho * self.mask
        out = np.empty_like(rho)
        d = self.d
        for k in range(d):
            L = d - k
            i = np.arange(L)
            out[i, i + k] = self.E[k] @ rho[i, i + k]
            if k > 0:
                out[i + k, i] = self.E[k] @ rho[i + k, i]
        return out


def default_steps(Gphi, Gc, Gh):
    return max(64, int(np.ceil(6 * max(Gphi, Gc, Gh))))


def propagate(psi0, beta, Gphi=0.0, Gc=0.0, Gh=0.0, steps=None, d=DIM):
    """rho(beta) and d rho/d beta after the noisy signal drive (Strang split)."""
    S = steps or default_steps(Gphi, Gc, Gh)
    a = ladder(d)
    G = a.T - a
    D = expm((beta / S) * G)
    full = NoiseStep(Gphi, Gc, Gh, 1.0 / S, d)
    half = NoiseStep(Gphi, Gc, Gh, 0.5 / S, d)
    rho = np.outer(psi0, psi0).astype(float)
    drho = np.zeros_like(rho)
    rho = half.apply(rho)
    for s in range(1, S + 1):
        rho = D @ rho @ D.T
        drho = D @ drho @ D.T + (G @ rho - rho @ G) / S
        ns = half if s == S else full
        rho = ns.apply(rho); drho = ns.apply(drho)
    return rho, drho


def qfi_from_C(C, dC):
    return abs(dC) ** 2 + np.real(np.conj(C) * dC) ** 2 / max(1 - abs(C) ** 2, 1e-300)


def qfi_from_bloch(r, dr):
    r = np.asarray(r); dr = np.asarray(dr)
    return dr @ dr + (r @ dr) ** 2 / max(1 - r @ r, 1e-300)


def qcrb_ideal_curve(rho, drho, alphas, d=DIM):
    out = []
    for al in alphas:
        V = displace_matrix(2j * al, d)
        out.append(1 / qfi_from_C(np.trace(V @ rho), np.trace(V @ drho)))
    return np.array(out)


# ---------------------------------------------------------------- closed form (eq. 33 / 74 / 106)
def laguerre(n, x):
    c = np.zeros(n + 1); c[n] = 1
    return lagval(x, c)


def qcrb_theory(al, n, r=0.0, Gc=0.0, Gh=0.0):
    K = Gc - Gh
    decay = np.exp(-K)
    diffusion = (Gc + Gh) if abs(K) < 1e-9 else (Gc + Gh) * (1 - decay) / K
    f = 1.0 - K / 4 if abs(K) < 1e-9 else (2 / K) * (1 - np.exp(-K / 2))
    B = diffusion + decay * np.exp(-2 * r)
    L = laguerre(n, 4 * al ** 2 * decay * np.exp(-2 * r))
    return np.exp(4 * al ** 2 * B) / (16 * al ** 2 * f ** 2 * L ** 2)


if __name__ == "__main__":
    al = np.arange(0.05, 2.6, 0.01)
    print("port vs closed form / 1113 saved outputs (beta = 2)")
    for (n, Gc, Gh, ref) in [(0, 0, 0, 0.67957), (1, 0, 0, 0.374884), (3, 0, 0, 0.256426),
                             (1, 1.0, 0, 1.99675), (1, 0.5, 0.2, 2.39218)]:
        rho, drho = propagate(fock(n), 2.0, Gc=Gc, Gh=Gh)
        q = qcrb_ideal_curve(rho, drho, al)
        th = min(qcrb_theory(a, n, Gc=Gc, Gh=Gh) for a in al)
        print(f"  n={n} Gc={Gc} Gh={Gh}: port {q.min():.6g}  theory {th:.6g}  1113 {ref}")
    for (Gp, Gc, Gh, ref) in [(1.0, 0, 0, 1.34845), (3.0, 0, 0, 5.42664), (1.0, 0.3, 0.1, 3.98036)]:
        rho, drho = propagate(fock(1), 2.0, Gp, Gc, Gh)
        print(f"  n=1 Gphi={Gp} Gc={Gc} Gh={Gh}: port {qcrb_ideal_curve(rho, drho, al).min():.6g}  1113 {ref}")
