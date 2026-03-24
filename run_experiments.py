"""
SynechismCore v19.0 — Main Experiment Runner
=============================================
Runs all 5 experiments across multiple seeds.
Reports honest results with correct p-values.

Usage:
    # Full run (all 5 experiments, 5 seeds, ~4-6 hours on Kaggle P100)
    python run_experiments.py

    # Quick test run (fewer seeds/epochs)
    python run_experiments.py --quick

    # Single experiment
    python run_experiments.py --experiment lorenz
    python run_experiments.py --experiment ks_pde
    python run_experiments.py --experiment finance
    python run_experiments.py --experiment weather
    python run_experiments.py --experiment robotics

Setup on Kaggle:
    1. New Notebook → Settings → Accelerator → GPU T4 x2 or P100
    2. In a code cell: !pip install torchdiffeq scipy -q
    3. Upload this zip, extract, run the command above

Author: Paul E. Harris IV — SynechismCore v19.0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import SynechismODE, FairTransformer, FairLSTM, count_parameters
from train import train_model, evaluate_model
from stats import compute_full_stats, aggregate_multi_seed, print_results_table
from data import (
    make_lorenz_dataset, make_ks_dataset,
    make_finance_dataset, make_weather_dataset,
    make_robotics_dataset
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN = 128
SEEDS  = [42, 0, 1, 7, 100]  # 5 seeds for multi-seed validation

LORENZ_RHO_TRAIN = [18., 20., 22., 24., 26., 28.]
LORENZ_RHO_TEST  = [35., 40., 45., 50.]


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_lorenz(seed: int, epochs: int = 150, n_traj: int = 120) -> dict:
    """Lorenz 63 bifurcation: train ρ∈{18-28}, test ρ∈{35,40,45,50}"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n  [Lorenz | seed={seed}] Generating data...")

    _, X_tr, Y_tr = make_lorenz_dataset(LORENZ_RHO_TRAIN, n_traj=n_traj, seed=seed)
    print(f"    Train: {X_tr.shape[0]:,} sequences")

    # Models
    ode  = SynechismODE(in_dim=3, out_dim=3, hidden=HIDDEN, pred_steps=Y_tr.shape[1])
    tf   = FairTransformer(in_dim=3, out_dim=3, hidden=HIDDEN, pred_steps=Y_tr.shape[1])
    lstm = FairLSTM(in_dim=3, out_dim=3, hidden=HIDDEN, pred_steps=Y_tr.shape[1])

    # Train
    train_model(ode,  X_tr, Y_tr, lr=1e-3, epochs=epochs, name="SynechismODE", device=DEVICE)
    train_model(tf,   X_tr, Y_tr, lr=6e-4, epochs=epochs, name="Transformer",  device=DEVICE)
    train_model(lstm, X_tr, Y_tr, lr=1e-3, epochs=epochs, name="LSTM",         device=DEVICE)

    # Evaluate on each test rho
    all_rho_results = {}
    for rho in LORENZ_RHO_TEST:
        _, X_te, Y_te = make_lorenz_dataset([rho], n_traj=40, seed=seed+999)
        ode_p,  true = evaluate_model(ode,  X_te, Y_te, device=DEVICE)
        tf_p,   _    = evaluate_model(tf,   X_te, Y_te, device=DEVICE)
        lstm_p, _    = evaluate_model(lstm, X_te, Y_te, device=DEVICE)

        r_tf   = compute_full_stats(ode_p, tf_p,   true, "ODE", "Transformer")
        r_lstm = compute_full_stats(ode_p, lstm_p, true, "ODE", "LSTM")
        all_rho_results[str(rho)] = {"vs_transformer": r_tf, "vs_lstm": r_lstm}

    # Aggregate across rho
    tf_ratios   = [all_rho_results[str(r)]["vs_transformer"]["ratio"] for r in LORENZ_RHO_TEST]
    lstm_ratios = [all_rho_results[str(r)]["vs_lstm"]["ratio"] for r in LORENZ_RHO_TEST]

    summary = {
        "experiment": "lorenz63",
        "seed": seed,
        "avg_ratio_vs_tf":   float(np.mean(tf_ratios)),
        "avg_ratio_vs_lstm": float(np.mean(lstm_ratios)),
        "ode_beats_tf_all":  all(all_rho_results[str(r)]["vs_transformer"]["ode_wins"] for r in LORENZ_RHO_TEST),
        "per_rho": all_rho_results,
        "params": {
            "ode": count_parameters(ode),
            "tf": count_parameters(tf),
            "lstm": count_parameters(lstm),
        }
    }

    print(f"    [Lorenz seed={seed}] ODE/TF avg: {summary['avg_ratio_vs_tf']:.3f}× | beats all: {summary['ode_beats_tf_all']}")
    return summary


def run_ks_pde(seed: int, epochs: int = 100, n_traj: int = 40) -> dict:
    """KS PDE bifurcation: train ν=1.0, test ν=0.5"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    N_KS = 64
    print(f"\n  [KS PDE | seed={seed}] Generating data...")

    _, X_tr, Y_tr = make_ks_dataset(nu=1.0, n_traj=n_traj, N=N_KS, seed=seed)
    _, X_te, Y_te = make_ks_dataset(nu=0.5, n_traj=20,     N=N_KS, seed=seed+500)
    print(f"    Train: {X_tr.shape[0]:,} | Test: {X_te.shape[0]:,}")

    ode  = SynechismODE(in_dim=N_KS, out_dim=N_KS, hidden=HIDDEN, pred_steps=Y_tr.shape[1],
                        solver='rk4')  # rk4 more stable for stiff KS
    tf   = FairTransformer(in_dim=N_KS, out_dim=N_KS, hidden=HIDDEN, pred_steps=Y_tr.shape[1])
    lstm = FairLSTM(in_dim=N_KS, out_dim=N_KS, hidden=HIDDEN, pred_steps=Y_tr.shape[1])

    train_model(ode,  X_tr, Y_tr, lr=1e-3, epochs=epochs, batch_size=32, name="SynechismODE", device=DEVICE)
    train_model(tf,   X_tr, Y_tr, lr=6e-4, epochs=epochs, batch_size=32, name="Transformer",  device=DEVICE)
    train_model(lstm, X_tr, Y_tr, lr=1e-3, epochs=epochs, batch_size=32, name="LSTM",         device=DEVICE)

    ode_p,  true = evaluate_model(ode,  X_te, Y_te, device=DEVICE)
    tf_p,   _    = evaluate_model(tf,   X_te, Y_te, device=DEVICE)
    lstm_p, _    = evaluate_model(lstm, X_te, Y_te, device=DEVICE)

    r_tf   = compute_full_stats(ode_p, tf_p,   true, "ODE", "Transformer")
    r_lstm = compute_full_stats(ode_p, lstm_p, true, "ODE", "LSTM")

    print_results_table(r_tf, f"KS PDE (seed={seed})")

    return {"experiment": "ks_pde", "seed": seed,
            "vs_transformer": r_tf, "vs_lstm": r_lstm,
            "params": {"ode": count_parameters(ode), "tf": count_parameters(tf)}}


def run_finance(seed: int, epochs: int = 80) -> dict:
    """Finance VIX regime: train calm, test crisis periods"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n  [Finance | seed={seed}] Generating data...")

    ds_tr, X_tr, Y_tr, _, X_te, Y_te = make_finance_dataset(seed=seed)
    print(f"    Train: {X_tr.shape[0]:,} | Test: {X_te.shape[0]:,}")

    in_dim = X_tr.shape[-1]  # 3 features
    ode  = SynechismODE(in_dim=in_dim, out_dim=in_dim, hidden=64, pred_steps=Y_tr.shape[1])
    tf   = FairTransformer(in_dim=in_dim, out_dim=in_dim, hidden=64, pred_steps=Y_tr.shape[1], nhead=4, nlayers=3)
    lstm = FairLSTM(in_dim=in_dim, out_dim=in_dim, hidden=64, pred_steps=Y_tr.shape[1])

    train_model(ode,  X_tr, Y_tr, lr=5e-4, epochs=epochs, batch_size=64, name="SynechismODE", device=DEVICE)
    train_model(tf,   X_tr, Y_tr, lr=3e-4, epochs=epochs, batch_size=64, name="Transformer",  device=DEVICE)
    train_model(lstm, X_tr, Y_tr, lr=5e-4, epochs=epochs, batch_size=64, name="LSTM",         device=DEVICE)

    ode_p,  true = evaluate_model(ode,  X_te, Y_te, device=DEVICE)
    tf_p,   _    = evaluate_model(tf,   X_te, Y_te, device=DEVICE)
    lstm_p, _    = evaluate_model(lstm, X_te, Y_te, device=DEVICE)

    r_tf   = compute_full_stats(ode_p, tf_p,   true, "ODE", "Transformer")
    r_lstm = compute_full_stats(ode_p, lstm_p, true, "ODE", "LSTM")

    print_results_table(r_tf, f"Finance (seed={seed})")

    return {"experiment": "finance", "seed": seed,
            "vs_transformer": r_tf, "vs_lstm": r_lstm}


def run_weather(seed: int, epochs: int = 100, n_traj: int = 30) -> dict:
    """Weather L96: train F∈{3,4,5,6}, test F∈{14,18,22}"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n  [Weather | seed={seed}] Generating data...")

    F_TRAIN = [3., 4., 5., 6.]
    F_TEST  = [14., 18., 22.]
    N_L96   = 40

    _, X_tr, Y_tr = make_weather_dataset(F_TRAIN, n_traj=n_traj, N=N_L96, seed=seed)
    _, X_te, Y_te = make_weather_dataset(F_TEST,  n_traj=15,     N=N_L96, seed=seed+300)
    print(f"    Train: {X_tr.shape[0]:,} | Test: {X_te.shape[0]:,}")

    ode  = SynechismODE(in_dim=N_L96, out_dim=N_L96, hidden=HIDDEN, pred_steps=Y_tr.shape[1], solver='rk4')
    tf   = FairTransformer(in_dim=N_L96, out_dim=N_L96, hidden=HIDDEN, pred_steps=Y_tr.shape[1])
    lstm = FairLSTM(in_dim=N_L96, out_dim=N_L96, hidden=HIDDEN, pred_steps=Y_tr.shape[1])

    train_model(ode,  X_tr, Y_tr, lr=1e-3, epochs=epochs, batch_size=32, name="SynechismODE", device=DEVICE)
    train_model(tf,   X_tr, Y_tr, lr=6e-4, epochs=epochs, batch_size=32, name="Transformer",  device=DEVICE)
    train_model(lstm, X_tr, Y_tr, lr=1e-3, epochs=epochs, batch_size=32, name="LSTM",         device=DEVICE)

    ode_p,  true = evaluate_model(ode,  X_te, Y_te, device=DEVICE)
    tf_p,   _    = evaluate_model(tf,   X_te, Y_te, device=DEVICE)
    lstm_p, _    = evaluate_model(lstm, X_te, Y_te, device=DEVICE)

    r_tf   = compute_full_stats(ode_p, tf_p,   true, "ODE", "Transformer")
    r_lstm = compute_full_stats(ode_p, lstm_p, true, "ODE", "LSTM")

    print_results_table(r_tf, f"Weather L96 (seed={seed})")

    return {"experiment": "weather", "seed": seed,
            "vs_transformer": r_tf, "vs_lstm": r_lstm}


def run_robotics(seed: int, epochs: int = 100, n_traj: int = 150) -> dict:
    """
    Robotics actuator failure: train γ=0.5, test γ=0.05

    NOTE: γ=0.05 is the honest test (not γ=0.0 which diverges for ALL models).
    The physics of near-zero damping is well-defined and ODE architectures
    should have an advantage because they can model the resonance dynamics
    via the continuous manifold.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n  [Robotics | seed={seed}] Train γ=0.5, Test γ=0.05")

    GAMMA_TRAIN = [0.5, 0.4, 0.3, 0.2]   # range of training damping
    GAMMA_TEST  = [0.05]                   # near-failure (honest test)

    _, X_tr, Y_tr = make_robotics_dataset(GAMMA_TRAIN, n_traj=n_traj, seed=seed)
    _, X_te, Y_te = make_robotics_dataset(GAMMA_TEST,  n_traj=50,     seed=seed+777)
    print(f"    Train: {X_tr.shape[0]:,} | Test: {X_te.shape[0]:,}")

    ode  = SynechismODE(in_dim=2, out_dim=2, hidden=64, pred_steps=Y_tr.shape[1])
    tf   = FairTransformer(in_dim=2, out_dim=2, hidden=64, pred_steps=Y_tr.shape[1], nhead=4, nlayers=3)
    lstm = FairLSTM(in_dim=2, out_dim=2, hidden=64, pred_steps=Y_tr.shape[1])

    train_model(ode,  X_tr, Y_tr, lr=1e-3, epochs=epochs, name="SynechismODE", device=DEVICE)
    train_model(tf,   X_tr, Y_tr, lr=6e-4, epochs=epochs, name="Transformer",  device=DEVICE)
    train_model(lstm, X_tr, Y_tr, lr=1e-3, epochs=epochs, name="LSTM",         device=DEVICE)

    ode_p,  true = evaluate_model(ode,  X_te, Y_te, device=DEVICE)
    tf_p,   _    = evaluate_model(tf,   X_te, Y_te, device=DEVICE)
    lstm_p, _    = evaluate_model(lstm, X_te, Y_te, device=DEVICE)

    r_tf   = compute_full_stats(ode_p, tf_p,   true, "ODE", "Transformer")
    r_lstm = compute_full_stats(ode_p, lstm_p, true, "ODE", "LSTM")

    print_results_table(r_tf, f"Robotics (seed={seed})")

    return {"experiment": "robotics", "seed": seed,
            "vs_transformer": r_tf, "vs_lstm": r_lstm}


# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_MAP = {
    "lorenz":   run_lorenz,
    "ks_pde":   run_ks_pde,
    "finance":  run_finance,
    "weather":  run_weather,
    "robotics": run_robotics,
}

EPOCHS_FULL  = {"lorenz": 150, "ks_pde": 100, "finance": 80, "weather": 100, "robotics": 100}
EPOCHS_QUICK = {"lorenz":  50, "ks_pde":  40, "finance": 30, "weather":  40, "robotics":  40}


def run_all(experiments=None, seeds=None, epochs_map=None, output_dir="./results"):
    if experiments is None:
        experiments = list(EXPERIMENT_MAP.keys())
    if seeds is None:
        seeds = SEEDS
    if epochs_map is None:
        epochs_map = EPOCHS_FULL

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    all_results = {}

    for exp_name in experiments:
        runner  = EXPERIMENT_MAP[exp_name]
        epochs  = epochs_map[exp_name]
        seed_results = []

        print(f"\n{'='*70}")
        print(f"  EXPERIMENT: {exp_name.upper()} | epochs={epochs} | seeds={seeds}")
        print(f"{'='*70}")

        for seed in seeds:
            r = runner(seed=seed, epochs=epochs)
            seed_results.append(r)

        # Aggregate
        tf_stats = [r["vs_transformer"] for r in seed_results]
        agg = aggregate_multi_seed(tf_stats)

        all_results[exp_name] = {
            "per_seed": seed_results,
            "aggregated": agg,
        }

        print(f"\n  ── {exp_name.upper()} SUMMARY ──")
        print(f"  ODE/TF ratio:  {agg['ratio_mean']:.3f}× ± {agg['ratio_std']:.3f}")
        print(f"  ODE wins all seeds: {agg['wins_all_seeds']}")
        print(f"  p-value range: [{agg['p_value_min']:.2e}, {agg['p_value_max']:.2e}]")

        # Save incrementally
        save_path = os.path.join(output_dir, f"{exp_name}_results.json")
        with open(save_path, "w") as f:
            json.dump(all_results[exp_name], f, indent=2, default=str)
        print(f"  Saved: {save_path}")

    # Final summary table
    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY — SynechismCore v19.0")
    print(f"  Seeds: {seeds}")
    print(f"{'='*70}")
    print(f"{'Experiment':<15} {'ODE/TF mean':>12} {'±std':>8} {'wins all':>10} {'p-min':>12}")
    print(f"{'-'*60}")

    for exp_name, res in all_results.items():
        agg = res["aggregated"]
        wins_str = "✅ YES" if agg["wins_all_seeds"] else "❌ NO"
        print(f"{exp_name:<15} {agg['ratio_mean']:>12.3f}× {agg['ratio_std']:>7.3f}  "
              f"{wins_str:>10}  {agg['p_value_min']:>12.2e}")

    # Save final summary
    summary_path = os.path.join(output_dir, "summary_v19.json")
    with open(summary_path, "w") as f:
        json.dump({
            "version": "19.0",
            "seeds": seeds,
            "runtime_minutes": (time.time() - t_start) / 60,
            "experiments": {k: v["aggregated"] for k, v in all_results.items()},
        }, f, indent=2, default=str)

    print(f"\n✅ All results saved to {output_dir}/")
    print(f"   Total runtime: {(time.time()-t_start)/60:.1f} min")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynechismCore v19.0 Experiments")
    parser.add_argument("--experiment", choices=list(EXPERIMENT_MAP.keys()) + ["all"],
                        default="all", help="Which experiment to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: fewer epochs for testing")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to use (default: [42,0,1,7,100])")
    parser.add_argument("--output", default="./results",
                        help="Output directory for results JSON")
    args = parser.parse_args()

    seeds   = args.seeds or SEEDS
    epochs  = EPOCHS_QUICK if args.quick else EPOCHS_FULL
    exps    = list(EXPERIMENT_MAP.keys()) if args.experiment == "all" else [args.experiment]

    print(f"\nSynechismCore v19.0 — Experiment Suite")
    print(f"{'='*70}")
    print(f"Experiments: {exps}")
    print(f"Seeds:       {seeds}")
    print(f"Mode:        {'QUICK' if args.quick else 'FULL'}")
    print(f"Device:      {DEVICE}")
    print(f"{'='*70}\n")

    run_all(experiments=exps, seeds=seeds, epochs_map=epochs, output_dir=args.output)
