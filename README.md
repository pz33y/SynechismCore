# SynechismCore
**Continuous-Time Neural Models for Regime-Shifting Dynamical Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-20.1-blue.svg)](https://github.com/pz33y/SynechismCore/releases)
[![Seeds](https://img.shields.io/badge/seeds-0,1,7,42,100-green.svg)](experiments/)
[![Compute](https://img.shields.io/badge/compute-Kaggle%20free%20GPU-orange.svg)](https://kaggle.com)

> **v20.1 — Hybrid architecture fully implemented.** The three components of Eq. 4 are real running code: continuous ODE brain, learned event detector, discrete jump function. KS PDE: **1.43× MAE over Transformer** (5 seeds, std ±0.003). Long-horizon coherence: **19,940 steps** (15.8× beyond LSTM).

**📖 Full Whitepaper & Website:**
- **[SynechismCore Website](https://synechism-jznepj9g.manus.space)** — Interactive research showcase with visualizations, metrics, and architecture details
- **[Full Whitepaper PDF (v20.1 Enhanced)](https://github.com/pz33y/SynechismCore/releases/download/v20.1/Synechism_v20_Whitepaper_Enhanced.pdf)** — 26 pages including philosophy, experiments, and enhanced graphics
- **[GitHub Release v20.1](https://github.com/pz33y/SynechismCore/releases/tag/v20.1)** — All assets and documentation

---

## What Is This?

Synechism is a stabilized Neural ODE for predicting chaotic dynamical systems across **bifurcations** — parameter changes that qualitatively alter system behavior. The core argument: **continuous architectures are structurally aligned with systems governed by differential equations** in a way that discrete Transformers and SSMs are not.

Named after C. S. Peirce's principle of synechism (1892): reality is fundamentally continuous.

---

## The Hybrid Architecture — Eq. 4 (Now Fully Implemented)

```
h_{t+1} = ODE_integrate(h_t, dt) + E(h_t) × [J(h_t) + C(h_t)]
```

| Symbol | Class | File | Role |
|--------|-------|------|------|
| **ODE** | `AttractorODEFunc` | `src/model.py` | Continuous brain — smooth physics |
| **E(h)** | `EventDetector` | `src/hyperagent.py` | Detects discontinuities, gates correction |
| **J(h)** | `JumpFunction` | `src/hyperagent.py` | Discrete jump, unconstrained magnitude |
| **C(h)** | `SmoothCorrection` | `src/hyperagent.py` | Near-discontinuity, Lipschitz-bounded |

Use `SynechismODE(..., hybrid=True)` to get the full hybrid model.

---

## Results

### Primary Confirmed Result

| Benchmark | Ratio | Seeds | Std | p-value |
|-----------|-------|-------|-----|---------|
| **KS PDE** (nu: 1.0→0.5) | **1.43×** | 5 | ±0.003 | <0.001 |

### Long-Horizon Coherence

| Model | Steps | vs LSTM |
|-------|-------|---------|
| ODE without stabilization | 960 | 0.76× ⚠️ |
| LSTM | 1,260 | 1.0× |
| Transformer | 1,840 | 1.46× |
| **Synechism v20.1** | **19,940** | **15.8×** |

---

## v20.1 Architecture Components

| Component | File | What it is |
|-----------|------|-----------|
| Attractor stabilization | `model.py` | `-alpha×(‖h‖²−R²)×h` — enables 19,940-step coherence |
| phi-scaling | `model.py` | `t_k=(k×phi) mod 1` — p=0.0000 vs uniform p=0.83 |
| SAGA goal vector | `model.py` | Learned directional prior — +4.0% contribution |
| HyperAgent | `hyperagent.py` | EventDetector + JumpFunction + SmoothCorrection |
| Mamba (ZOH) | `model.py` | Fair baseline with correct discretization |

---

## Reproduce

```bash
git clone https://github.com/pz33y/SynechismCore.git
cd SynechismCore && pip install -r requirements.txt

# Quick test
python run_experiments.py --experiment ks_pde --seeds 42 --epochs 30
```

---

## Citation

```bibtex
@misc{harris2026synechism,
  title  = {SynechismCore v20.1: Continuous-Time Neural Models for Regime-Shifting Dynamical Systems},
  author = {Harris IV, Paul E.},
  year   = {2026},
  url    = {https://github.com/pz33y/SynechismCore}
}
```

MIT License · Paul E. Harris IV · 2026
