# SynechismCore v23.0 

**Latent Neural ODEs with Aperiodic φ-Scaling for Chaotic Dynamical Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Target: arXiv](https://img.shields.io/badge/Target-arXiv%20%7C%20ML4PS%20%40%20NeurIPS%202026-green.svg)]()
[![Status: Benchmark Pending](https://img.shields.io/badge/v23%20Benchmark-Pending%20H100%20Run-yellow.svg)]()

> **Author:** Paul E. Harris IV · Independent Researcher, Mashantucket Pequot Nation  
> **Status:** Codebase locked, mathematically patched, and optimized for Hopper GPUs (v23.0.1). Pending full 10-seed empirical benchmark run.

## Overview
SynechismCore is a Latent Neural ODE architecture that uses golden-ratio ($\varphi$) aperiodic time sampling and attractor regularization to model chaotic and regime-shifting dynamical systems. The core question this project answers empirically:

> **When do continuous latent dynamics provide a measurable advantage over discrete sequence models — and when do they not?**

We report results honestly. The architecture's confirmed wins (KS-PDE, Lorenz bifurcation) are documented alongside its failures (Robotics actuator loss). The **v23.0 architecture** introduces three surgical components to directly address the failure modes identified in v22.

---

## 🚀 What's New in v23.0 (Architecture Complete)

The v23 architecture is fully implemented and patched for mathematical correctness and hardware efficiency. 

1. **IrrationalShutter:** Replaces adaptive `dopri5` stepping with forced `rk4` steps exactly on the $\varphi$-Weyl sequence grid. *Goal: Eliminate resonance and phase-locking on high-chaos fractal attractors (e.g., Lorenz $\rho \ge 45$).*
2. **ElasticManifold:** Couples a learned regime-shift event detector to the attractor radius constraint, allowing the topological boundary to dynamically "breathe" (expand via a GELU network). *Goal: Recover the $0.52\times$ loss on discontinuous underdamped physics (Robotics).*
3. **LaminarBypass:** A local-curvature heuristic that routes smooth dynamics through a computationally cheap, $\Delta t$-aware linear projection, reserving heavy ODE integration only for turbulent bursts. *Goal: Drastically reduce wall-clock time.*

---

## 📊 Current Status: Confirmed Results vs Pending Claims

**Confirmed v22 Results (Produced on Kaggle free-tier P100/T4):**
*   ✅ **KS-PDE ($\nu$: 1.0→0.5):** $1.43\times$ MAE win over Transformer across 5 seeds.
*   ✅ **Lorenz-63 Coherence:** 19,940 continuous prediction steps before divergence ($15.8\times$ vs baselines).
*   ✅ **$\varphi$-Significance:** Aperiodic sampling achieves $p=0.0000$ over uniform grids.
*   ❌ **Robotics ($\gamma$: 0.5→0.05):** ODE loses badly ($0.52\times$) to Transformer due to fixed attractor sphere.
*   ⚠️ **Weather L96 / Finance:** Statistically tied or marginally significant.

**Pending v23 Benchmark (Awaiting H100 execution):**
*   ⏳ **Claim 4 Ablation:** Does $\varphi$ specifically beat $\sqrt{2}$ and $e$?
*   ⏳ **Robotics Recovery:** Does `ElasticManifold` turn the $0.52\times$ loss into a win?
*   ⏳ **High-Chaos Lorenz:** Does `IrrationalShutter` prevent topological collapse at $\rho=50$?
*   ⏳ **KS-PDE 10-Seed Validation:** Does the $1.43\times$ headline hold under rigorous 10-seed Mann-Whitney U testing?

*(Note: Empirical JSON results will be pushed to `/results/v23/` upon run completion.)*

---

## ⚙️ Repository Structure

```text
SynechismCore/
├── src/
│   ├── models.py             # v22 core: SynechismV20, all baselines
│   ├── v23_components.py     # ElasticManifold, IrrationalShutter, LaminarBypass
│   ├── hyperagent.py         # Event detector + discrete jump correction
│   ├── data.py               # 5 synthetic dynamical system generators
│   ├── train.py              # Training loop + evaluation module
│   ├── stats.py              # Corrected Mann-Whitney U significance testing
│   ├── chaotic_metrics.py    # VPT, sMAPE, nRMSE, fractal dimension
│   ├── quantum_lattice.py    # PhiLattice, FibonacciLattice, HaltonLattice
│   └── hyevo.py              # Evolutionary hyperparameter search
│
├── launch_h100.py            # H100-optimized full suite launcher (TF32 + torch.compile)
├── run_v23_benchmark.py      # v23 vs v22 head-to-head + coherence testing
├── run_phi_ablation.py       # φ vs √2 vs e ablation 
├── run_experiments.py        # Legacy v22 5-experiment benchmark
└── results/                  # JSON outputs go here
```

## 💻 Quick Start & Reproduction

1. **Install Dependencies**
```bash
pip install torch torchdiffeq scipy pandas matplotlib
```
(PyTorch 2.0+ required for torch.compile optimizations).

2. **Sanity Check (~5 mins)**
Verify the pipeline works before renting heavy compute:
```bash
python launch_h100.py --quick
```

3. **Run the Full v23 Benchmark Suite**
This executes the $\varphi$-ablation, v23 component tests, coherence tests, and KS-PDE multi-seed validation. Optimized for NVIDIA Hopper (H100) or Ampere (A100) GPUs.
```bash
# Start a persistent screen session
screen -S synechism

# Launch the full 10-seed suite
python launch_h100.py --seeds 0 1 2 3 4 5 6 7 8 9
```

🔬 **Significance Testing Methodology**
All multi-model comparisons use the non-parametric Mann-Whitney U test over per-sample error distributions to preserve the i.i.d. assumption:
```python
# Averages error over Time and Dimension FIRST
ode_errors = np.abs(ode_preds - true_values).mean(axis=(1, 2))   # Shape: (N,)
baseline_errors = np.abs(baseline_preds - true_values).mean(axis=(1, 2))
stat, p = scipy.stats.mannwhitneyu(ode_errors, baseline_errors, alternative='less')
```
This prevents the artificial inflation of $N$ (and subsequently hacked $p$-values) common in flattened time-series benchmarking.

📜 **Citation**
```bibtex
@misc{harris2026synechismcore,
  title     = {SynechismCore: Latent Neural ODEs with Aperiodic phi-Scaling for Chaotic Dynamical Systems},
  author    = {Harris IV, Paul E.},
  year      = {2026},
  note      = {Preprint. v23 benchmark pending.},
  url       = {https://github.com/pz33y/SynechismCore}
}
```

License: MIT
