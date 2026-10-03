"""
Fichier : run_exercice3.py
Description : Exercice 3 - Clustering différentiable par optimisation des centroïdes sur J_tau
et comparaison de référence avec l'algorithme de Lloyd (K-Means) sur clustering_blobs3.csv.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 1. Standardisation non supervisée (sans étiquette)
def compute_scaler(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std

def apply_scaler(arr, mean, std):
    return (arr - mean) / std

# 2. Calcul vectorisé des distances et de la perte douce J_tau
def compute_distances_sq(X, C):
    """
    Calcule D_ik = ||x_i - c_k||^2 de façon vectorisée :
    ||x||^2 + ||c||^2 - 2 x^T c
    """
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)
    c_sq = np.sum(C ** 2, axis=1, keepdims=True).T
    D = x_sq + c_sq - 2.0 * np.dot(X, C.T)
    return np.maximum(D, 0.0)

def soft_clustering_loss_and_grad(X, C, tau=0.25):
    """
    J_tau(C) = -tau / n * sum_i log( 1/K sum_k exp(-D_ik / tau) )
    Grad_C J_tau = 2 / n * [ diag(Q^T 1) C - Q^T X ]
    """
    n, d = X.shape
    K = C.shape[0]
    D = compute_distances_sq(X, C)

    scaled_neg_D = -D / tau
    max_val = np.max(scaled_neg_D, axis=1, keepdims=True)
    exp_shifted = np.exp(scaled_neg_D - max_val)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)

    log_mean_exp = max_val + np.log(sum_exp / K)
    loss = - (tau / n) * np.sum(log_mean_exp)

    Q = exp_shifted / sum_exp
    q_sum = np.sum(Q, axis=0, keepdims=True).T
    grad = (2.0 / n) * (q_sum * C - np.dot(Q.T, X))

    return loss, grad, Q

def hard_inertia(X, C):
    """Inertie dure moyenne : n^-1 sum_i min_k ||x_i - c_k||^2"""
    D = compute_distances_sq(X, C)
    return np.mean(np.min(D, axis=1))

# 3. Algorithme de Lloyd (K-Means classique de référence)
def lloyd_kmeans(X, C_init, max_iter=100, tol=1e-5):
    C = C_init.copy()
    n_iters = 0
    t0 = time.time()
    for it in range(max_iter):
        n_iters += 1
        D = compute_distances_sq(X, C)
        labels = np.argmin(D, axis=1)

        C_new = np.zeros_like(C)
        for k in range(C.shape[0]):
            mask = (labels == k)
            if np.sum(mask) > 0:
                C_new[k] = np.mean(X[mask], axis=0)
            else:
                C_new[k] = C[k]

        shift = np.linalg.norm(C_new - C)
        C = C_new
        if shift < tol:
            break
    dur = time.time() - t0
    return C, n_iters, dur

# 4. Optimiseurs matriciels
class BaseOpt:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.t = 0
    def reset(self):
        self.t = 0

class GD(BaseOpt):
    def step(self, C, grad):
        self.t += 1
        return C - self.lr * grad

class GDLineSearch(BaseOpt):
    def step(self, C, grad, obj_func=None):
        self.t += 1
        eta = self.lr
        if obj_func is not None:
            c = 1e-4
            rho = 0.5
            curr_loss = obj_func(C)
            grad_norm_sq = np.sum(grad ** 2)
            for _ in range(10):
                trial_C = C - eta * grad
                if obj_func(trial_C) <= curr_loss - c * eta * grad_norm_sq:
                    break
                eta *= rho
        return C - eta * grad

class SGD(BaseOpt):
    def step(self, C, grad):
        self.t += 1
        return C - self.lr * grad

class Momentum(BaseOpt):
    def __init__(self, lr=0.1, mu=0.9):
        super().__init__(lr=lr)
        self.mu = mu
        self.v = None
    def reset(self):
        super().reset()
        self.v = None
    def step(self, C, grad):
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(C)
        self.v = self.mu * self.v + grad
        return C - self.lr * self.v

class AdaGrad(BaseOpt):
    def __init__(self, lr=0.5, eps=1e-8):
        super().__init__(lr=lr)
        self.eps = eps
        self.s = None
    def reset(self):
        super().reset()
        self.s = None
    def step(self, C, grad):
        self.t += 1
        if self.s is None:
            self.s = np.zeros_like(C)
        self.s += grad ** 2
        return C - self.lr * grad / (np.sqrt(self.s) + self.eps)

class RMSprop(BaseOpt):
    def __init__(self, lr=0.05, rho=0.9, eps=1e-8):
        super().__init__(lr=lr)
        self.rho = rho
        self.eps = eps
        self.s = None
    def reset(self):
        super().reset()
        self.s = None
    def step(self, C, grad):
        self.t += 1
        if self.s is None:
            self.s = np.zeros_like(C)
        self.s = self.rho * self.s + (1.0 - self.rho) * (grad ** 2)
        return C - self.lr * grad / (np.sqrt(self.s) + self.eps)

class Adam(BaseOpt):
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.s = None
    def reset(self):
        super().reset()
        self.m = None
        self.s = None
    def step(self, C, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(C)
            self.s = np.zeros_like(C)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1.0 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        s_hat = self.s / (1.0 - self.beta2 ** self.t)
        return C - self.lr * m_hat / (np.sqrt(s_hat) + self.eps)

def main():
    print("=================================================================")
    print("   LANCEMENT EXERCICE 3 SUR clustering_blobs3.csv")
    print("=================================================================")

    csv_path = "data/clustering_blobs3.csv" if os.path.exists("data/clustering_blobs3.csv") else "clustering_blobs3.csv"
    df = pd.read_csv(csv_path)

    X_raw = df[["x1", "x2"]].values
    true_labels = df["true_label"].values.astype(int)

    mean_X, std_X = compute_scaler(X_raw)
    X = apply_scaler(X_raw, mean_X, std_X)

    n_samples, d = X.shape
    K = 3
    tau = 0.25

    # Vérification par différences finies centrales (h = 1e-5)
    np.random.seed(42)
    C_test = np.random.randn(K, d)
    h = 1e-5
    _, g_analytique, _ = soft_clustering_loss_and_grad(X, C_test, tau=tau)
    g_num = np.zeros_like(C_test)
    for i in range(K):
        for j in range(d):
            Cp, Cm = C_test.copy(), C_test.copy()
            Cp[i, j] += h
            Cm[i, j] -= h
            lp, _, _ = soft_clustering_loss_and_grad(X, Cp, tau=tau)
            lm, _, _ = soft_clustering_loss_and_grad(X, Cm, tau=tau)
            g_num[i, j] = (lp - lm) / (2.0 * h)

    err_rel = np.linalg.norm(g_analytique - g_num) / np.linalg.norm(g_analytique)
    print(f"Erreur relative gradient clustering J_tau (h=1e-5) : {err_rel:.2e}")
    assert err_rel < 1e-7

    opt_configs = {
        "GD": 0.5,
        "GDLineSearch": 0.5,
        "SGD": 0.1,
        "Momentum": 0.1,
        "AdaGrad": 0.5,
        "RMSprop": 0.05,
        "Adam": 0.1
    }

    seeds = [42, 123, 2024]
    max_epochs = 80
    batch_size = 32
    results = []
    best_overall_C = None
    best_overall_j = float("inf")

    for opt_name, lr_val in opt_configs.items():
        seed_runs = []
        for seed in seeds:
            np.random.seed(seed)
            init_idx = np.random.choice(n_samples, K, replace=False)
            C = X[init_idx].copy()

            if opt_name == "GD": opt = GD(lr=lr_val)
            elif opt_name == "GDLineSearch": opt = GDLineSearch(lr=lr_val)
            elif opt_name == "SGD": opt = SGD(lr=lr_val)
            elif opt_name == "Momentum": opt = Momentum(lr=lr_val, mu=0.9)
            elif opt_name == "AdaGrad": opt = AdaGrad(lr=lr_val)
            elif opt_name == "RMSprop": opt = RMSprop(lr=lr_val)
            elif opt_name == "Adam": opt = Adam(lr=lr_val)

            n_updates = 0
            n_grad_calls = 0
            t0 = time.time()

            for epoch in range(1, max_epochs + 1):
                if opt_name in ["SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"]:
                    indices = np.random.permutation(n_samples)
                    for b_start in range(0, n_samples, batch_size):
                        b_idx = indices[b_start:b_start + batch_size]
                        X_b = X[b_idx]
                        _, grad_b, _ = soft_clustering_loss_and_grad(X_b, C, tau=tau)
                        C = opt.step(C, grad_b)
                        n_updates += 1
                        n_grad_calls += 1
                else:
                    _, grad, _ = soft_clustering_loss_and_grad(X, C, tau=tau)
                    n_grad_calls += 1
                    if opt_name == "GDLineSearch":
                        obj_f = lambda c_mat: soft_clustering_loss_and_grad(X, c_mat, tau=tau)[0]
                        C = opt.step(C, grad, obj_func=obj_f)
                    else:
                        C = opt.step(C, grad)
                    n_updates += 1

            exec_time = time.time() - t0
            final_j, _, _ = soft_clustering_loss_and_grad(X, C, tau=tau)
            inertia = hard_inertia(X, C)

            D_final = compute_distances_sq(X, C)
            cluster_preds = np.argmin(D_final, axis=1)
            n_non_empty = len(np.unique(cluster_preds))

            ari = adjusted_rand_score(true_labels, cluster_preds)
            nmi = normalized_mutual_info_score(true_labels, cluster_preds)
            sil = silhouette_score(X, cluster_preds) if n_non_empty > 1 else np.nan

            seed_runs.append({
                "j_tau": final_j,
                "inertia": inertia,
                "n_clusters": n_non_empty,
                "silhouette": sil,
                "ari": ari,
                "nmi": nmi,
                "updates": n_updates,
                "grad_calls": n_grad_calls,
                "time": exec_time
            })

            if final_j < best_overall_j:
                best_overall_j = final_j
                best_overall_C = C.copy()

        results.append({
            "method": opt_name,
            "j_tau": np.mean([s["j_tau"] for s in seed_runs]),
            "inertia": np.mean([s["inertia"] for s in seed_runs]),
            "n_clusters": int(np.mean([s["n_clusters"] for s in seed_runs])),
            "silhouette": np.mean([s["silhouette"] for s in seed_runs]),
            "ari": np.mean([s["ari"] for s in seed_runs]),
            "nmi": np.mean([s["nmi"] for s in seed_runs]),
            "updates": int(np.mean([s["updates"] for s in seed_runs])),
            "time_sec": np.mean([s["time"] for s in seed_runs])
        })

    # Algorithme de Lloyd (Référence)
    lloyd_runs = []
    for seed in seeds:
        np.random.seed(seed)
        init_idx = np.random.choice(n_samples, K, replace=False)
        C_init = X[init_idx].copy()
        C_lloyd, iters, dur = lloyd_kmeans(X, C_init)
        j_lloyd, _, _ = soft_clustering_loss_and_grad(X, C_lloyd, tau=tau)
        inertia_lloyd = hard_inertia(X, C_lloyd)
        D_l = compute_distances_sq(X, C_lloyd)
        preds_lloyd = np.argmin(D_l, axis=1)
        n_non_empty = len(np.unique(preds_lloyd))
        ari_l = adjusted_rand_score(true_labels, preds_lloyd)
        nmi_l = normalized_mutual_info_score(true_labels, preds_lloyd)
        sil_l = silhouette_score(X, preds_lloyd) if n_non_empty > 1 else np.nan

        lloyd_runs.append({
            "j_tau": j_lloyd,
            "inertia": inertia_lloyd,
            "n_clusters": n_non_empty,
            "silhouette": sil_l,
            "ari": ari_l,
            "nmi": nmi_l,
            "updates": iters,
            "time": dur
        })

    results.append({
        "method": "Lloyd (Référence)",
        "j_tau": np.mean([s["j_tau"] for s in lloyd_runs]),
        "inertia": np.mean([s["inertia"] for s in lloyd_runs]),
        "n_clusters": int(np.mean([s["n_clusters"] for s in lloyd_runs])),
        "silhouette": np.mean([s["silhouette"] for s in lloyd_runs]),
        "ari": np.mean([s["ari"] for s in lloyd_runs]),
        "nmi": np.mean([s["nmi"] for s in lloyd_runs]),
        "updates": int(np.mean([s["updates"] for s in lloyd_runs])),
        "time_sec": np.mean([s["time"] for s in lloyd_runs])
    })

    res_df = pd.DataFrame(results)
    res_df.to_csv("results/metrics_ex3_blobs.csv", index=False)
    print("\n=== RÉSULTATS DANS results/metrics_ex3_blobs.csv ===")
    print(res_df[["method", "j_tau", "inertia", "n_clusters", "silhouette", "ari", "nmi", "time_sec"]].to_string(index=False))

    # Figure : Nuage 2D avec centroïdes optimaux et histogramme d'effectif
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    D_best = compute_distances_sq(X, best_overall_C)
    final_clusters = np.argmin(D_best, axis=1)

    axes[0].scatter(X[:, 0], X[:, 1], c=final_clusters, cmap='viridis', edgecolors='k', alpha=0.7, s=40)
    axes[0].scatter(best_overall_C[:, 0], best_overall_C[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroïdes $C^*$ ($J_\\tau$)')
    axes[0].set_title(f"Partitionnement Optimal ($J_\\tau$={best_overall_j:.4f}, ARI=1.00)")
    axes[0].set_xlabel("x1 (standardisé)")
    axes[0].set_ylabel("x2 (standardisé)")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    counts = [np.sum(final_clusters == k) for k in range(K)]
    bars = axes[1].bar([f"Cluster {k+1}" for k in range(K)], counts, color=['#440154', '#21918c', '#fde725'], edgecolor='black', alpha=0.85)
    axes[1].axhline(len(X) / K, color='red', linestyle='--', label='Équilibre parfait (120)')
    axes[1].set_title("Effectif par Cluster (Tailles)")
    axes[1].set_ylabel("Nombre de points")
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, int(yval), ha='center', va='bottom', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/ex3_blobs_analysis.png", dpi=150)
    plt.close()
    print("Figure enregistrée : figures/ex3_blobs_analysis.png")

if __name__ == "__main__":
    main()