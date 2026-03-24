# Changelog — SynechismCore

## [v19.0] — 2026-03-24 · CURRENT

### Fixed
- **CRITICAL: p-value bug** — v18.5 produced p=0.50/1.00 for all tests (comparing scalars). Fixed to Mann-Whitney U over per-sample error distributions (N≥1000 samples). Real significance now computed.
- **Lorenz regression** — v18.5 added U-Net encoder that interfered with ODE attractor term, causing 0.94× (loss). Reverted to clean GRU encoder from v17.2 that produced 1.25–1.33×.
- **Robotics test invalid** — γ=0.0 (exact zero damping) causes energy divergence for ALL models. Fixed to γ=0.05 (near-failure, physically valid, honest test).
- **v19 code was placeholder** — previous v19 zip had `# placeholder` content. Now contains real working code.

### Architecture
- Clean ODE design: GRU encoder → h0 projection → AttractorODEFunc → decoder
- No U-Net, no skip connections (these hurt Lorenz in v18.5)
- Spectral norm on nonlinear branch retained
- φ-scaling on ODE time points retained
- SAGA goal vector retained

### Removed
- EEG 88% brain correlation claim (no methodology exists)
- Consciousness probability claims (not a scientific claim)
- "Holographic 5D Memory" (not implemented)
- "Cognitive Privacy RLWE" (not implemented)
- "Pre-linguistic agency" (not a measurable property)

### Results (confirmed)
- KS PDE: 1.43× over Transformer, 5 seeds, ±0.003

---

## [v18.5] — 2026-03 · SUPERSEDED

- Added 5 experiments (Lorenz, KS, Finance, Weather, Robotics)
- Added Mamba baseline
- Added Neural CDE baseline
- BROKEN: p-values all 0.50/1.00 (statistical test bug)
- BROKEN: Lorenz regressed to 0.94× (U-Net encoder interference)
- KS PDE result was real: 1.43× confirmed

---

## [v17.2] — 2026-03 · SUPERSEDED

- ODE beats Transformer on Lorenz: 1.18–1.33× at ρ∈{35,40,45,50}, p<1e-43
- 19,940-step coherence (15.8× vs LSTM)
- φ-scaling proven: p=0.0000 vs p=0.83 uniform
- Removed EEG/consciousness claims (were in v16.7)

---

## [v16.7] — 2026-02 · HISTORICAL

- Contained unverified EEG 88% claim and consciousness probability
- These claims are permanently removed from all subsequent versions
