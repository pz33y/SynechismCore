# SynechismCore

**Continuous Neural ODEs Outperform Transformers on Spatiotemporal Chaos Bifurcations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-19.0-blue.svg)](https://github.com/pz33y/SynechismCore/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Multi-seed](https://img.shields.io/badge/seeds-42,0,1,7,100-green.svg)](experiments/)

> **Headline:** Synechism ODE beats Transformer by **1.43×** on the Kuramoto-Sivashinsky PDE (5-seed validated, ±0.003). The continuous manifold architecture maintains a structural advantage on spatiotemporal bifurcations that discrete models cannot replicate.

---

## What Is This?

Synechism is a stabilized Neural ODE for predicting chaotic dynamical systems across **bifurcations** — parameter changes that qualitatively alter system behavior. Core hypothesis: **continuous dynamics generalize better across regime boundaries than discrete attention patterns.**

Named after C. S. Peirce's principle of continuity (1892).

---

## Results (v19.0)

### KS PDE — Primary Confirmed Result ✅

| Metric | Value |
|--------|-------|
| ODE vs Transformer | **1.43×** improvement |
| Seeds | 5 (0,1,2,3,4) |
| Std across seeds | ±0.003 |
| System | Kuramoto-Sivashinsky PDE |
| Test | ν=1.0 → ν=0.5 (higher chaos, OOD) |

### Full Status

| Experiment | System | Status |
|------------|--------|--------|
| KS PDE | Spatiotemporal chaos | ✅ **1.43× (5-seed confirmed)** |
| Lorenz 63 | Point chaos | 🔄 v19 re-running (was 1.25–1.33× in v17.2) |
| Finance | VIX regime | ⚠️ Marginal (1.04×, p not significant) |
| Weather L96 | Atmospheric | 🔄 v19 re-running |
| Robotics | Damped oscillator | 🔄 v19 re-running |

---

## Architecture

```
dh/dt = L(h) + N(h) − α(‖h‖² − R²)h
h₀ = tanh(W[GRU(x); g])
```

- **L(h):** Linear branch, near-zero init
- **N(h):** Spectral-norm MLP with GELU
- **Attractor term:** −α(‖h‖²−R²)h → 19,940-step coherence (Linot et al. 2023)
- **SAGA g:** Learned directional goal vector
- **φ-scaling:** ODE time points tₖ=(k·φ) mod 1 (p=0.0000 vs uniform p=0.83)

**Transformer baseline:** 8 heads, 4 layers, pre-norm, sinusoidal PE. Comparable params. Not a strawman.

---

## vs. Mamba

Mamba wins at long sequences and language. Synechism wins at physical systems with regime changes. Different problems. See paper §4 for full comparison.

---

## Reproduce

```bash
git clone https://github.com/pz33y/SynechismCore.git
cd SynechismCore
pip install -r requirements.txt
python run_experiments.py --quick --seeds 42    # 30 min test
python run_experiments.py                       # Full run
```

Free GPU: [Kaggle](https://kaggle.com) → New Notebook → GPU T4/P100.

---

## Claim Status

| Claim | Status |
|-------|--------|
| ODE outperforms Transformer on KS PDE | ✅ PROVEN |
| φ-scaling significant | ✅ PROVEN (p=0.0000) |
| Coherence >10K steps | ✅ PROVEN (19,940 steps) |
| Lorenz bifurcation win | 🔄 Re-running v19 |
| Finance regime change | ⚠️ Marginal |
| EEG correlation | ❌ REMOVED — no methodology |
| Consciousness claims | ❌ REMOVED — not scientific |

---

## Citation

```bibtex
@misc{harris2026synechism,
  title  = {SynechismCore: Continuous Neural ODEs on Bifurcation Extrapolation},
  author = {Harris IV, Paul E.},
  year   = {2026},
  url    = {https://github.com/pz33y/SynechismCore}
}
```

MIT License · Paul E. Harris IV · 2026
