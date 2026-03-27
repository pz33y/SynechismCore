# Changelog — SynechismCore

## [v20.0] — 2026-03-26 · CURRENT

### Added
- **HyperAgent** (`src/hyperagent.py`) — EventDetector, JumpFunction, SmoothCorrection
- **HyEvo** (`src/hyevo.py`) — Multi-island evolutionary optimizer for alpha, R, phi_base
- **UFO Encoder** — U-Net Conv1d with skip connections and variational bottleneck
- **Koopman Lifting** — phi-dimensional expansion with gated projection
- **Fourier Skip Connection** — sin/cos frequency features in AttractorODEFunc
- **Quantum Lattice** (`src/quantum_lattice.py`) — Phi, Fibonacci, and Halton sequences
- **ZOH Mamba baseline** — correct dA=exp(dt×A) discretization
- **make_synechism() factory** — 5 variants: base, phi, skip, full, hybrid
- **Eq. 4 fully implemented** — all three hybrid components are running code

### Fixed
- **p-value bug** (was all 0.50/1.00) — now Mann-Whitney U over per-sample distributions
- **quantum_lattice.py syntax error** — unterminated string literal on line 249
- **src/__init__.py** — unclosed docstring

### Architecture
- `SynechismV20` unified class replaces separate `SynechismODE` / `SynechismHybrid`
- Hybrid variant (Eq. 4) used for robotics and finance experiments
- Full variant (UFO + Koopman + Fourier, no HyperAgent) used for KS PDE, Lorenz, Weather

---

## [v19.0] — 2026-03-25 · Superseded

- Added HyperAgent and HyEvo (first implementations, less complete)
- Fixed Lorenz architecture regression from v18.5
- Fixed p-value bug
- KS PDE 1.43× confirmed across 5 seeds

---

## [v18.5] — 2026-03 · Superseded

- 5 experiments with Mamba and Neural CDE baselines
- BROKEN: p-values all 0.50/1.00
- BROKEN: Lorenz regressed to 0.94× (U-Net encoder interference)
- KS PDE result was real: 1.43× confirmed

---

## [v17.2] — 2026-03 · Historical reference

- ODE beats Transformer on Lorenz: 1.25-1.33× at ρ∈{35,40,45,50}, p<1e-43 (seed=42)
- 19,940-step coherence (15.8× vs LSTM)
- phi-scaling: p=0.0000 vs p=0.83 uniform
- Removed EEG/consciousness claims

---

## [v16.7] — 2026-02 · Historical

- Contained unverified EEG 88% and consciousness probability claims
- Permanently removed in v17.2+
