"""Does replacing the laser SDF readout by the ideal unitary D(+-i alpha) change the QCRB?

Three readout models acting on the same post-signal motional state rho_m (and d rho_m / d beta):
  ideal : U = exp(-i alpha sigma_x (a + a^dag))                       (what 1104 and 1113 use)
  rwa   : bichromatic red+blue sideband laser, exact exp(i eta x) coupling, RWA w.r.t. trap frequency
  full  : same laser with carrier, counter-rotating terms and exact time dependence (no RWA)

Units: time in 1/g with g = eta*Omega/2 (the nominal SDF rate), so alpha = g t.
The ion starts in |g>, an equal superposition of the sigma_x eigenstates, as in 1104.
"""
import sys
import time
import numpy as np
from scipy.linalg import expm, eigh
from engine import DIM, ladder, displace_matrix, fock, propagate, qcrb_ideal_curve, qfi_from_bloch

# ---------------------------------------------------------------- 1104 physical parameters
hbar = 1.0545718e-34; amu = 1.66053906e-27; m = 39.96259 * amu
lam = 729.147e-9                      # Ca-40 S1/2 - D5/2
nu_hz = 1.9e6                         # axial trap frequency (1104: trap_freq = 19e5)
eta = (2 * np.pi / lam) / np.sqrt(2) * np.sqrt(hbar / (2 * m * 2 * np.pi * nu_hz))
t_sig = 4.413e-3                      # 1113 cell 13 saved output, I = 1e3, beta = 2
g_rad = 2 * 2.0 / t_sig / 2           # g = eta*Omega/2 = beta / t_sig  [rad/s]
Omega_hz = 2 * g_rad / eta / (2 * np.pi)

d = DIM
a = ladder(d); X = a + a.T
I2 = np.eye(2); Id = np.eye(d)
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1.0, -1.0]).astype(complex)
sp = np.array([[0, 0], [1, 0]], complex)        # |e><g|, basis (g, e)
Deta = displace_matrix(1j * eta, d)             # D(i eta), padded then cropped
Ksub = np.diag(np.diag(Deta, -1), -1)           # <n+1|D|n>  (adds a phonon)
Ksup = np.diag(np.diag(Deta, 1), 1)             # <n-1|D|n>
K = Ksub + Ksup                                 # resonant part of D(i eta e^{i nu t}) e^{-+ i nu t}


def bloch(rho_tot):
    rs = rho_tot.reshape(2, d, 2, d).trace(axis1=1, axis2=3)
    return np.real([np.trace(rs @ s) for s in (sx, sy, sz)])


def qcrb_from_evolution(rho_m, drho_m, Us):
    sg = np.diag([1.0, 0.0])
    R0 = np.kron(sg, rho_m).astype(complex); dR0 = np.kron(sg, drho_m).astype(complex)
    out = []
    for U in Us:
        r = bloch(U @ R0 @ U.conj().T); dr = bloch(U @ dR0 @ U.conj().T)
        out.append(1 / qfi_from_bloch(r, dr))
    return np.array(out)


def U_ideal(alphas):
    w, V = eigh(np.kron(sx, X).real)
    return [V @ np.diag(np.exp(-1j * w * al)) @ V.conj().T for al in alphas]


def U_rwa(alphas):
    # H/g = (1/eta) * ( -i sigma_+ K + i sigma_- K^dag ),  -> sigma_x (a + a^dag) as eta -> 0
    H = (1 / eta) * (-1j * np.kron(sp, K) + 1j * np.kron(sp.conj().T, K.conj().T))
    w, V = eigh(H)
    return [V @ np.diag(np.exp(-1j * w * al)) @ V.conj().T for al in alphas]


def U_full(alphas, Omega_over_nu, per_period=24):
    """Exact time-dependent laser Hamiltonian (carrier + both sidebands + counter-rotating, no RWA).
    H(t)/g = -i (Omega/g) cos(nu t) sigma_+ (x) M(t) + h.c.,  M(t) = D(i eta e^{i nu t})."""
    Om = 2 / eta                                  # Omega in units of g
    nu = Om / Omega_over_nu
    dt_target = 2 * np.pi / nu / per_period
    jk = np.subtract.outer(np.arange(d), np.arange(d))
    Us, U = [], np.eye(2 * d, dtype=complex)
    t = 0.0
    for al in alphas:
        nstep = max(1, int(np.ceil((al - t) / dt_target)))
        dt = (al - t) / nstep
        for _ in range(nstep):
            tm = t + dt / 2
            M = np.exp(1j * nu * tm * jk) * Deta
            Hp = -1j * Om * np.cos(nu * tm) * np.kron(sp, M)
            H = Hp + Hp.conj().T
            U = expm(-1j * H * dt) @ U
            t += dt
        Us.append(U.copy())
    return Us


def report(label, q_ref, q):
    i_ref, i = np.argmin(q_ref), np.argmin(q)
    win = q_ref <= 3 * q_ref.min()                 # alphas where the readout is worth using
    rel = np.max(np.abs(q - q_ref)[win] / q_ref[win])
    print(f"    {label:<34s} min QCRB {q[i]:.5g} at alpha {alphas[i]:.2f}"
          f"  (ideal {q_ref[i_ref]:.5g} at {alphas[i_ref]:.2f});  min shift {q[i]/q_ref[i_ref]-1:+.2e};"
          f"  max rel. diff where QCRB < 3x min {rel:.2e}")
    return rel


if __name__ == "__main__":
    print(f"eta = {eta:.4f}, nu/2pi = {nu_hz/1e6:.2f} MHz, g/2pi = {g_rad/2/np.pi:.1f} Hz, "
          f"Omega/2pi = {Omega_hz/1e3:.2f} kHz, Omega/nu = {Omega_hz/nu_hz:.2e}")
    alphas = np.round(np.arange(0.1, 2.21, 0.05), 4)
    Ui, Ur = U_ideal(alphas), U_rwa(alphas)
    cases = [("noiseless", 0, 0, 0), ("1104 point: 365/50/10 Hz", 1.611, 0.221, 0.0441), ("dephasing Gphi t = 3", 3.0, 0, 0)]
    for n in (0, 1, 3):
        for (lab, Gp, Gc, Gh) in cases:
            rho, drho = propagate(fock(n), 2.0, Gp, Gc, Gh)
            qi = qcrb_from_evolution(rho, drho, Ui)
            qe = qcrb_ideal_curve(rho, drho, alphas)
            win = qe <= 3 * qe.min()
            print(f"n={n}, {lab}  (top-5 Fock population {np.trace(rho[-5:, -5:]):.1e};"
                  f" Bloch vs C path, ideal readout: {np.max(np.abs(qi-qe)[win]/qe[win]):.1e})")
            report("laser, exact eta, RWA", qi, qcrb_from_evolution(rho, drho, Ur))
    sys.stdout.flush()

    # carrier / counter-rotating terms: inflate Omega to make them visible, then read off the scaling
    print("\nfull time-dependent laser (no RWA) vs RWA laser, n = 1, 1104 noise point")
    rho, drho = propagate(fock(1), 2.0, 1.611, 0.221, 0.0441)
    qr = qcrb_from_evolution(rho, drho, Ur)
    for ratio in (0.1, 0.05, 0.025):
        t0 = time.time()
        qf = qcrb_from_evolution(rho, drho, U_full(alphas, ratio))
        w = qr <= 3 * qr.min()
        print(f"  Omega/nu = {ratio:<6}: max rel. diff vs RWA laser (QCRB < 3x min) {np.max(np.abs(qf-qr)[w]/qr[w]):.2e};"
              f"  min shift {qf.min()/qr.min()-1:+.2e}"
              f"   ({time.time()-t0:.0f} s)")
        sys.stdout.flush()
    qf2 = qcrb_from_evolution(rho, drho, U_full(alphas, 0.1, per_period=48))
    qf1 = qcrb_from_evolution(rho, drho, U_full(alphas, 0.1, per_period=24))
    w = qf2 <= 3 * qf2.min()
    print(f"  time-step check at Omega/nu = 0.1 (24 vs 48 steps/period): {np.max(np.abs(qf2-qf1)[w]/qf2[w]):.1e}")
