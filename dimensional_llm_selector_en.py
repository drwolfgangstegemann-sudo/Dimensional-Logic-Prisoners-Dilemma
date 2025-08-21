
"""
Dimensional Logic LLM Selector
==============================

This module applies Dimensional Logic (σ₂: systematic derivation, 
μ₃: reflexivity, κ₄: context coherence) to re-rank Large Language 
Model (LLM) outputs. 

Idea:
- Instead of choosing outputs only by probability (as in standard LLMs),
  we add two epistemic dimensions:
    μ₃ → reflexive alignment of outputs with user expectations
    κ₄(θ) → contextual coherence (social, semantic, or task-specific)

Usage:
1. Provide a list of candidate outputs from an LLM.
2. Define reflexivity scores (μ₃) and context scores (κ₄).
3. Call `evaluate_outputs` to compute the dimensional utility.
4. Select the output with the highest score.

This proof-of-concept illustrates how Dimensional Logic can enrich 
language model reasoning beyond classical likelihoods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Example candidate outputs from an LLM
# ----------------------------------------------------------
outputs = [
    "The cooperation of agents leads to higher stability.",
    "Defection dominates unless external enforcement is present.",
    "Cooperation can emerge if trust and norms are integrated."
]

# Reflexivity scores μ₃: How much the output reflects back on itself / user intent
mu_scores = [0.6, 0.4, 0.9]

# Context scores κ₄(θ): How coherent the output is with context or norms
kappa_scores = [0.7, 0.3, 0.85]

# ----------------------------------------------------------
# Evaluation function: Dimensional Utility
# ----------------------------------------------------------
def evaluate_outputs(outputs, mu, kappa, alpha=1.0, beta=1.0, R=3, T=5):
    """
    Compute dimensional utilities for LLM outputs.

    Args:
        outputs (list): Candidate text outputs.
        mu (list): Reflexivity scores μ₃ for each output.
        kappa (list): Context coherence scores κ₄ for each output.
        alpha (float): Weight for reflexivity.
        beta (float): Weight for context coherence.
        R (float): Baseline reward.
        T (float): Temptation threshold for rational dominance.

    Returns:
        DataFrame: Outputs with calculated dimensional utilities.
    """
    utilities = []
    for i, out in enumerate(outputs):
        U = R + alpha * mu[i] + beta * kappa[i]
        utilities.append((out, mu[i], kappa[i], U, U >= T))

    df = pd.DataFrame(utilities, columns=["Output", "μ₃ (Reflexivity)", "κ₄ (Context)", "Utility", "≥T?"])
    return df

# ----------------------------------------------------------
# Example run
# ----------------------------------------------------------
df_results = evaluate_outputs(outputs, mu_scores, kappa_scores, alpha=1.2, beta=1.0)
print(df_results)

# ----------------------------------------------------------
# Visualization: How cooperation-like responses are favored
# ----------------------------------------------------------
def visualize_selector(mu, kappa, alpha_range=(0,2), beta_range=(0,2)):
    alphas = np.linspace(alpha_range[0], alpha_range[1], 20)
    betas = np.linspace(beta_range[0], beta_range[1], 20)

    coop_utilities = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            U = 3 + a * mu[-1] + b * kappa[-1]  # last output = cooperative
            coop_utilities[i,j] = U

    plt.figure(figsize=(7,5))
    plt.contourf(betas, alphas, coop_utilities, cmap="viridis")
    plt.colorbar(label="Utility of cooperative output")
    plt.xlabel("β (context weight)")
    plt.ylabel("α (reflexivity weight)")
    plt.title("Dimensional Logic Utility for LLM Outputs")
    plt.show()

visualize_selector(mu_scores, kappa_scores)
