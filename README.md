# sdf-adversarial-gmm

Clean, reproducible codebase for comparing Soft-Penalty Neural SDFs and Adversarial GMM SDFs in asset pricing.

---

## Paper

Soft-Penalty Neural SDFs vs. Adversarial GMM:  
A Structural Comparison in Asset Pricing

Taeha Park (2025)

Paper (PDF): paper.pdf  
arXiv: https://arxiv.org/abs/7098952

---

## Abstract

This paper compares two neural approaches to estimating stochastic discount factors (SDFs) in asset pricing: 
the widely used soft-penalty Neural SDF and an adversarial generalized method of moments (Adversarial GMM) formulation. 
While the soft-penalty objective enforces the Euler equation only on average, the adversarial framework identifies 
worst-case pricing-error directions through a critic network, closely mirroring optimal GMM instrumentation.

In controlled CCAPM simulations, both methods perform similarly. In contrast, empirical results using the 
Open Asset Pricing dataset show that Adversarial GMM produces substantially lower SDF volatility and more 
economically disciplined pricing behavior. The findings highlight the importance of adversarial moment 
selection for robust and interpretable SDF estimation in high-dimensional asset markets.

---

## Key Contributions

- Implements Soft-Penalty Neural SDF and Adversarial GMM SDF within a unified and reproducible framework.
- Interprets the adversarial critic as selecting worst-case pricing-error directions, providing a structural link to optimal GMM instruments.
- Demonstrates how adversarial moment selection affects SDF volatility, identification, and macro–firm risk decomposition.
- Provides both simulation and large-scale empirical evidence using the Open Asset Pricing dataset.

---

## Repository Structure

- `README.md`  
  Project overview and usage instructions.

- `requirements.txt`  
  Python dependencies required to reproduce the experiments.

- `scripts/`  
  Entry-point scripts for running simulations and empirical analyses.

- `src/`  
  Core reusable modules for SDF models, adversarial GMM training, and data preprocessing.


## Quick Start (Local)

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
