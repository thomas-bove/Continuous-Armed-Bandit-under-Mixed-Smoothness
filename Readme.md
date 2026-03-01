# Continuous-Armed Bandit under Mixed Smoothness

Reduction of a continuous bandit problem over `[0,1]^d` to a finite Multi-Armed Bandit via **Smolyak sparse-grid discretization**, with provable sublinear regret independent of dimension.

---

## Problem

Standard MAB assumes a finite, known set of arms. In many real-world settings (hyperparameter tuning, dose optimization, resource allocation), the action space is continuous and high-dimensional.

The naive approach — discretizing `[0,1]^d` with a uniform grid — requires `O(ε^{-d})` points to achieve approximation error `ε`, making it intractable for `d > 2` (curse of dimensionality).

**This work exploits a structural assumption** — *dominating mixed smoothness (DMS)* — to achieve dimension-independent regret.

---

## Key Idea

If the reward function `f` has bounded mixed weak derivatives (i.e., `f ∈ W¹_mix([0,1]^d)`), then a Smolyak sparse-grid approximant built from `N` nodes satisfies:
```
‖f − f̃_N‖ ≤ C · N⁻¹ · (ln N)^β
```

This rate is **independent of `d`** (dimension enters only through the log factor `β = d−1`), compared to `O(N^{-1/d})` for isotropic Lipschitz functions.

**Strategy:** restrict the agent to the `N` Smolyak grid nodes → run UCB1 on the resulting `N`-armed bandit.

---

## Regret Bound

The cumulative regret decomposes as:
```
R(T) = T · b_N  +  R_MAB(T)
          ↑               ↑
  discretization      UCB1 learning
      bias            O(√NT log T)
```

Choosing `N = Θ(T^{1/3})` balances both terms, yielding:
```
R(T) = Õ(T^{2/3})
```

**The exponent 2/3 does not depend on d** — in contrast to Lipschitz bandits where the exponent `(d+1)/(d+2) → 1` as `d → ∞`.

| Setting | Regret |
|---|---|
| Lipschitz bandit, `d=2` | `Θ(T^{3/4})` |
| Lipschitz bandit, `d=5` | `Θ(T^{6/7})` |
| **DMS bandit (this work), any `d`** | **`Õ(T^{2/3})`** |

Full proof in [`theory/bandit_analysis.pdf`](theory/bandit_analysis.pdf).

---

## Repository Structure
```
├── README.md
├── requirements.txt
├── theory/
│   └── bandit_analysis.pdf
├── src/
│   ├── __init__.py
│   ├── sparse_grid.py        # Smolyak grid construction
│   └── ucb1.py               # UCB1 on finite arm set
└── experiments/
    ├── environment.py        # LinearBanditEnv, DMSBanditEnv
    ├──run_experiment.py     # Three experiments
    └── results/plots/
        ├── exp1_exponent.png
        ├── exp2_levels.png
        └── exp3_dms.png

```

---

## Reproduce
```bash
pip install -r requirements.txt
python experiments/run_experiment.py
# Plots saved to results/plots/
```

---

## Experiments

### Exp 1 — Regret exponent (linear reward, fixed grid)

`d=5`, `T=20000`, Smolyak level 3 (N=61 arms), linear reward `f(x) = θᵀx`.

![Exp 1](results/plots/exp1_exponent.png)

Estimated exponent: **0.811** (log-log fit after warmup).

> With N fixed and T growing, the optimal arm may not lie in the grid — the exponent drifts above 2/3 toward 1 (linear regret regime). This is consistent with theory: the 2/3 bound requires `N = Θ(T^{1/3})` to grow with T. Exp 3 validates this with a scaled grid.

---

### Exp 2 — Grid resolution vs. regret (linear and DMS)

`d=5`, `T=5000`, Smolyak level 2 (N=11) vs. level 3 (N=61).

![Exp 2](results/plots/exp2_levels.png)

**Linear reward:** both levels give near-identical regret — the linear function is well-approximated even by the coarse grid.

**DMS nonlinear reward:** level 2 (N=11) achieves lower regret exponent (0.26) vs. level 3 (0.40). With T=5000, the optimal grid size is `T^{1/3} ≈ 17` — the coarser grid is closer to optimal, consistent with theory.

---

### Exp 3 — Scaled grid, DMS reward

`d=5`, `T=10000`, `N` scaled as `T^{1/3}`.

![Exp 3](results/plots/exp3_dms.png)

Estimated exponent: **0.372**. The regret curve visibly flattens after `t ≈ 6000`, confirming sublinear growth and validating the `Õ(T^{2/3})` bound in the nonlinear DMS setting.

---

## Requirements
```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

---

## References

- Smolyak (1963) — sparse grid quadrature
- Bungartz & Griebel (2004) — Sparse grids, Acta Numerica
- Auer, Cesa-Bianchi, Fischer (2002) — UCB1 finite-time analysis
- Kleinberg (2005) — continuum-armed bandit lower bounds
- Temlyakov (2018) — Multivariate Approximation