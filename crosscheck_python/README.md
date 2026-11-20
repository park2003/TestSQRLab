# Python cross-check of the 1113 effective model

Independent of Julia. Needs `numpy` and `scipy`. These produced the "expected results" quoted in `1120_ionsim_check.ipynb`.

| Script | What it does |
| --- | --- |
| `engine.py` | Port of the 1113 engine (Strang split, exact noise step, exact ∂/∂β). `python3 engine.py` reproduces the 1113 benchmark tables. |
| `laser_check.py` | Replaces the ideal readout D(±iα) with a bichromatic sideband laser: exact e^{iηx} coupling (RWA), and the full time-dependent Hamiltonian with carrier and counter-rotating terms at inflated Ω/ν. |
| `readout_noise.py` | Lets the motional noise act during a finite-time readout of rate s·β (s = 1 is the 1104 timing). |

Run from this folder: `python3 laser_check.py` (about 1 min), `python3 readout_noise.py` (about 20 s).
