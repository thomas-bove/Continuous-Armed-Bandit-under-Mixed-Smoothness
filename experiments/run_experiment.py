import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.sparse_grid import smolyak_grid
from src.ucb1 import UCB1
from experiments.environment import LinearBanditEnv, DMSBanditEnv

def find_max_level(d, target_N):
    """Trova il max_level tale che la griglia abbia circa target_N punti."""
    for level in range(1, 8):
        grid = smolyak_grid(d, level)
        if len(grid) >= target_N:
            return level, len(grid)
    return 7, len(smolyak_grid(d, 7))

def run_experiment(d=5, T=5000, mode="fixed", env_type="linear", max_level=3):
    """
    mode='fixed'  → griglia fissa con max_level (come prima)
    mode='scaled' → N = round(T^(1/3)), max_level scelto di conseguenza
    env_type='linear' o 'dms'
    """
    if env_type == "linear":
        env = LinearBanditEnv(d)
    else:
        env = DMSBanditEnv(d)

    if mode == "scaled":
        target_N = max(4, round(T ** (1/3)))
        level, actual_N = find_max_level(d, target_N)
        print(f"  T={T}, target N={target_N}, actual N={actual_N}, level={level}")
    else:
        level = max_level

    grid = smolyak_grid(d, level)
    agent = UCB1(grid)

    cumulative_regret = 0
    regrets = []
    for t in range(T):
        arm_index = agent.select_arm()
        x = grid[arm_index]
        reward = env.reward(x)
        agent.update(arm_index, reward)
        regret = env.f_star - env.theta @ x if env_type == "linear" else env.f_star - env._f(x)
        cumulative_regret += regret
        regrets.append(cumulative_regret)

    return regrets

if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)
    d = 5

    # ── Esperimento 1: fit esponente (N fisso, funzione lineare) ──────────
    print("Esperimento 1: fit esponente")
    regrets = run_experiment(d=d, T=20000, mode="fixed", env_type="linear", max_level=3)
    log_T = np.log(np.arange(1, 20001))
    log_R = np.log(np.maximum(np.array(regrets), 1e-10))
    n_arms = len(smolyak_grid(d, 3))
    start = min(n_arms * 5, 500) 
    alpha, _ = np.polyfit(log_T[start:], log_R[start:], 1)
    print(f"  (fit iniziato a t={start}, N={n_arms} bracci)")
    print(f"  Esponente stimato: {alpha:.3f}  (atteso ≈ 0.667 per T^2/3)")

    plt.figure()
    plt.plot(regrets, label=f"esponente stimato={alpha:.3f}")
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.title(f"Exp 1 — Regret cumulativo (d={d}, N fisso)")
    plt.xlabel("Time"); plt.ylabel("Regret"); plt.legend()
    plt.savefig("results/plots/exp1_exponent.png", dpi=150, bbox_inches='tight')

    # ── Esperimento 2: N scalato con T vs N fisso ─────────────────────────
    # Sostituisci Esperimento 2 con questo

    print("Esperimento 2: confronto level=2 vs level=3")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, env_type in zip(axes, ["linear", "dms"]):
     for level, color in zip([2, 3], ["blue", "orange"]):
        N = len(smolyak_grid(d, level))
        r = run_experiment(d=d, T=5000, mode="fixed", env_type=env_type, max_level=level)
        log_T_ = np.log(np.arange(1, 5001))
        log_R_ = np.log(np.maximum(np.array(r), 1e-10))
        start = N * 5
        a, _ = np.polyfit(log_T_[start:], log_R_[start:], 1)
        ax.plot(r, color=color, label=f"level={level}, N={N}, exp={a:.2f}")
    ax.set_title(f"Exp 2 — {env_type} (d={d})")
    ax.set_xlabel("Time"); ax.set_ylabel("Regret"); ax.legend()

    plt.tight_layout()
    plt.savefig("results/plots/exp2_levels.png", dpi=150, bbox_inches='tight') 

    # ── Esperimento 3: funzione DMS non lineare ───────────────────────────
    print("Esperimento 3: funzione DMS non lineare")
    regrets_dms = run_experiment(d=d, T=2000, mode="scaled", env_type="dms")
    log_T2 = np.log(np.arange(1, 2001))
    log_R2 = np.log(np.maximum(np.array(regrets_dms), 1e-10))
    alpha2, _ = np.polyfit(log_T2[100:], log_R2[100:], 1)
    print(f"  Esponente stimato (DMS): {alpha2:.3f}")

    plt.figure()
    plt.plot(regrets_dms, label=f"esponente={alpha2:.3f}")
    plt.title(f"Exp 3 — Regret con funzione DMS (d={d})")
    plt.xlabel("Time"); plt.ylabel("Regret"); plt.legend()
    plt.savefig("results/plots/exp3_dms.png", dpi=150, bbox_inches='tight')

    plt.show()
    print("Fatto. Grafici salvati in results/plots/")
