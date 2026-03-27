"""
SynechismCore Statistical Interpreter — interpret_results.py
==========================================================
Implements the Mann-Whitney U test for per-sample absolute error distributions (N >= 1,000).
Provides robust, non-parametric validation for OOD bifurcation tasks.

Author: Paul E. Harris IV — SynechismCore v20.1
"""

import numpy as np
from scipy import stats

def analyze_significance(model_errors, baseline_errors, alpha=0.05):
    """
    Performs Mann-Whitney U test to compare error distributions.
    
    Args:
        model_errors: List/array of absolute errors from Synechism.
        baseline_errors: List/array of absolute errors from baseline (e.g., Transformer).
        alpha: Significance level.
        
    Returns:
        p_value, significant_flag, improvement_ratio
    """
    
    # Perform test
    stat, p_val = stats.mannwhitneyu(model_errors, baseline_errors, alternative='less')
    
    # Calculate improvement ratio (median errors)
    median_model = np.median(model_errors)
    median_baseline = np.median(baseline_errors)
    improvement_ratio = median_baseline / median_model if median_model > 0 else float('inf')
    
    significant = p_val < alpha
    
    return p_val, significant, improvement_ratio

def format_report(results):
    """Formats the statistical report for the whitepaper."""
    print("--- SynechismCore Statistical Report ---")
    for name, data in results.items():
        p, sig, ratio = data
        sig_str = "SIGNIFICANT" if sig else "NOT SIGNIFICANT"
        print(f"{name}: p={p:.4e} | {sig_str} | Improvement: {ratio:.2f}x")

if __name__ == "__main__":
    # Example validation for KS PDE
    np.random.seed(42)
    synechism_err = np.random.normal(0.2952, 0.003, 1000)
    transformer_err = np.random.normal(0.4207, 0.005, 1000)
    
    p, sig, ratio = analyze_significance(synechism_err, transformer_err)
    format_report({"KS PDE": (p, sig, ratio)})
