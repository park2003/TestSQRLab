"""Noise during a finite-time SDF readout.

1104 and 1113 apply the readout D(+-i alpha) instantly. A real SDF at rate g_R lasts t_R = alpha / g_R.
In engine units (t_sig = 1, signal rate beta) a readout s times faster than the signal drive gives
t_R = alpha / (s beta); s = 1 is the 1104 timing assumption (t_sig = 2 beta / (eta Omega), same laser).

Only the spin-coherence block rho_{+-} is needed: H = g sigma_x (a + a^dag) gives
  d rho_{+-}/dt = -i g (X rho + rho X) + L_noise rho_{+-},   rho_{+-}(0) = rho_m / 2,   C = 2 Tr rho_{+-}.
Integrated with a symmetric (Strang) split and the exact noise step; populations of |+>,|-> stay 1/2.
"""
import sys
import numpy as np
from scipy.linalg import expm
from engine import DIM, ladder, fock, propagate, qcrb_ideal_curve, NoiseStep, qfi_from_C


def readout_curve(rho_m, drho_m, alphas, beta, s, Gphi, Gc, Gh, sub=8, d=DIM):
    """QCRB(alpha) with noise acting during a readout of rate s*beta (engine units)."""
    a = ladder(d); X = a + a.T
    dal = alphas[1] - alphas[0]
    assert np.allclose(np.diff(alphas), dal) and np.isclose(alphas[0] / dal, round(alphas[0] / dal))
    da = dal / sub                                  # readout displacement per Strang step
    dt = da / (s * beta)                            # its duration in units of t_sig
    V = expm(-1j * da * X)
    half = NoiseStep(Gphi, Gc, Gh, dt / 2, d)
    B = 0.5 * rho_m.astype(complex); dB = 0.5 * drho_m.astype(complex)
    out, al = [], 0.0
    targets = set(np.round(alphas, 10))
    nsteps = int(round(alphas[-1] / da))
    for k in range(1, nsteps + 1):
        B = half.apply(B); dB = half.apply(dB)
        B = V @ B @ V; dB = V @ dB @ V
        B = half.apply(B); dB = half.apply(dB)
        al = round(k * da, 10)
        if al in targets:
            out.append(1 / qfi_from_C(2 * np.trace(B), 2 * np.trace(dB)))
    return np.array(out)


if __name__ == "__main__":
    beta = 2.0
    alphas = np.round(np.arange(0.05, 2.01, 0.05), 4)

    # checks: no noise -> identical to instantaneous readout; step halving
    rho, drho = propagate(fock(1), beta)
    q0 = qcrb_ideal_curve(rho, drho, alphas)
    q1 = readout_curve(rho, drho, alphas, beta, 1, 0, 0, 0)
    w = q0 <= 3 * q0.min()
    print(f"check, no noise: finite-time vs instantaneous readout  max rel. diff {np.max(np.abs(q1-q0)[w]/q0[w]):.1e}")
    rho, drho = propagate(fock(1), beta, 1.611, 0.221, 0.0441)
    qa = readout_curve(rho, drho, alphas, beta, 1, 1.611, 0.221, 0.0441, sub=8)
    qb = readout_curve(rho, drho, alphas, beta, 1, 1.611, 0.221, 0.0441, sub=16)
    w = qb <= 3 * qb.min()
    print(f"check, step halving at the 1104 point: max rel. diff {np.max(np.abs(qa-qb)[w]/qb[w]):.1e}\n")
    sys.stdout.flush()

    cases = [("1104 point 365/50/10 Hz", 1.611, 0.221, 0.0441), ("dephasing Gphi t = 1", 1.0, 0, 0),
             ("dephasing Gphi t = 3", 3.0, 0, 0), ("cooling Gc t = 0.5", 0, 0.5, 0)]
    print(f"{'case':<26}{'n':>3}  {'instant':>9} {'a*':>5} | {'s=1':>9} {'a*':>5} {'x':>6} | {'s=10':>9} {'a*':>5} {'x':>6}")
    for (lab, Gp, Gc, Gh) in cases:
        for n in (0, 1, 3):
            rho, drho = propagate(fock(n), beta, Gp, Gc, Gh)
            qi = qcrb_ideal_curve(rho, drho, alphas)
            row = f"{lab:<26}{n:>3}  {qi.min():9.4g} {alphas[qi.argmin()]:5.2f}"
            for s in (1, 10):
                q = readout_curve(rho, drho, alphas, beta, s, Gp, Gc, Gh)
                row += f" | {q.min():9.4g} {alphas[q.argmin()]:5.2f} {q.min()/qi.min():6.2f}"
            print(row); sys.stdout.flush()
