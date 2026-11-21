"""Does a laser-driven signal (instead of the ideal drive beta (a^dag - a)) change QCRB(alpha)?

A laser force acting on a spin eigenstate displaces the motion with the exact Lamb-Dicke coupling
<n+1|e^{i eta (a + a^dag)}|n> instead of eta sqrt(n+1). In the RWA the signal generator becomes
  G_eta = (1/eta) [ Im D(i eta)_{n+1,n} ] (sub-diagonal) - transpose,   G_eta -> a^dag - a as eta -> 0,
a real antisymmetric matrix, so the 1113 Strang engine runs unchanged with G -> G_eta.
Carrier / counter-rotating terms are left out: 1120 measured them at 6e-4 even at Omega/nu = 0.025,
and the 1104 laser has Omega/nu = 1.5e-3.
"""
import numpy as np
from scipy.linalg import expm
from engine import DIM, ladder, displace_matrix, fock, NoiseStep, default_steps, qcrb_ideal_curve

eta = 0.0497                                   # 1120 printout
a = ladder(DIM)
De = displace_matrix(1j * eta, DIM)
sub = np.diag(np.imag(np.diag(De, -1)) / eta, -1)
G_eta = sub - sub.T
G_ideal = a.T - a


def propagate_G(psi0, beta, Gp, Gc, Gh, G, steps=None):
    S = steps or default_steps(Gp, Gc, Gh)
    D = expm((beta / S) * G)
    full = NoiseStep(Gp, Gc, Gh, 1.0 / S); half = NoiseStep(Gp, Gc, Gh, 0.5 / S)
    rho = np.outer(psi0, psi0); drho = np.zeros_like(rho)
    rho = half.apply(rho)
    for s in range(1, S + 1):
        rho = D @ rho @ D.T
        drho = D @ drho @ D.T + (G @ rho - rho @ G) / S
        ns = half if s == S else full
        rho = ns.apply(rho); drho = ns.apply(drho)
    return rho, drho


if __name__ == "__main__":
    print(f"eta = {eta}; coupling ratio <n+1|G_eta|n> / sqrt(n+1) at n = 0, 5, 10, 20:",
          ", ".join(f"{G_eta[n+1, n]/np.sqrt(n+1):.4f}" for n in (0, 5, 10, 20)))
    al = np.round(np.arange(0.05, 2.61, 0.01), 4)
    cases = [("noiseless", 0, 0, 0), ("1104 point", 1.611, 0.221, 0.0441), ("dephasing 1", 1.0, 0, 0), ("dephasing 3", 3.0, 0, 0)]
    print(f"\n{'case':<12}{'n':>3} | {'ideal signal':>13} {'a*':>5} | {'laser signal':>13} {'a*':>5} | {'min shift':>9} | {'max diff, QCRB<3x min':>22}")
    for lab, Gp, Gc, Gh in cases:
        for n in (0, 1, 3):
            q0 = qcrb_ideal_curve(*propagate_G(fock(n), 2.0, Gp, Gc, Gh, G_ideal), al)
            q1 = qcrb_ideal_curve(*propagate_G(fock(n), 2.0, Gp, Gc, Gh, G_eta), al)
            w = q0 <= 3 * q0.min()
            print(f"{lab:<12}{n:>3} | {q0.min():13.5g} {al[q0.argmin()]:5.2f} | {q1.min():13.5g} {al[q1.argmin()]:5.2f} |"
                  f" {q1.min()/q0.min()-1:+8.2%} | {np.max(np.abs(q1-q0)[w]/q0[w]):21.2%}")
