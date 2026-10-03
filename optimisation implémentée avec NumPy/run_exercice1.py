"""
Fichier : run_exercice1.py
Description : Code autonome tout-en-un pour l'Exercice 1 sur 'regression_nonlinear.csv'.
Intègre :
- Standardisation calculée sur Train uniquement (et remise à l'échelle de y)
- Modèles OLS, Ridge et Ridge à Noyau RBF
- Les 7 optimiseurs (GD, GDLineSearch, SGD, Momentum, AdaGrad, RMSprop, Adam)
- Évaluation sur 3 graines fixées : [42, 123, 2024]
- Métriques et figures obligatoires générées automatiquement
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------
# 1. SCALER & NOYAU RBF (Vectorisés)
# -------------------------------------------------------------
def compute_scaler(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    if np.isscalar(std) or std.ndim == 0:
        std = 1.0 if std == 0.0 else float(std)
    else:
        std = np.where(std == 0.0, 1.0, std)
    return mean, std

def apply_scaler(arr, mean, std):
    return (arr - mean) / std

def invert_scaler(arr, mean, std):
    return arr * std + mean

def rbf_kernel_matrix(X1, X2, gamma):
    sq_norm1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq_norm2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
    dist_sq = sq_norm1 + sq_norm2 - 2.0 * np.dot(X1, X2.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.exp(-gamma * dist_sq)

def evaluate_metrics(y_true, y_pred):
    residual = y_true - y_pred
    mse = np.mean(residual ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(residual ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    return mse, rmse, mae, r2

# -------------------------------------------------------------
# 2. MODÈLES DE RÉGRESSION
# -------------------------------------------------------------
class LinearRegressionModel:
    def __init__(self, lmbda=0.0):
        self.lmbda = lmbda

    def _get_P(self, d):
        P = np.eye(d)
        P[0, 0] = 0.0  # Biais w0 non pénalisé
        return P

    def loss(self, w, X, y):
        n = X.shape[0]
        d = X.shape[1]
        residual = X @ w - y
        P = self._get_P(d)
        return np.sum(residual ** 2) / (2.0 * n) + 0.5 * self.lmbda * np.dot(w, P @ w)

    def gradient(self, w, X, y):
        n = X.shape[0]
        d = X.shape[1]
        residual = X @ w - y
        P = self._get_P(d)
        return (X.T @ residual) / n + self.lmbda * (P @ w)

    def hessian(self, X):
        n = X.shape[0]
        d = X.shape[1]
        P = self._get_P(d)
        return (X.T @ X) / n + self.lmbda * P

    def closed_form_solution(self, X, y):
        H = self.hessian(X)
        rhs = (X.T @ y) / X.shape[0]
        return np.linalg.solve(H, rhs)

class KernelRidgeRegressionModel:
    def __init__(self, lmbda=1e-3, sigma=1.0):
        self.lmbda = lmbda
        self.sigma = sigma
        self.gamma = 1.0 / (2.0 * (sigma ** 2))

    def loss(self, a, K, y):
        n = K.shape[0]
        residual = K @ a - y
        return np.sum(residual ** 2) / (2.0 * n) + 0.5 * self.lmbda * np.dot(a, K @ a)

    def gradient(self, a, K, y):
        n = K.shape[0]
        residual = K @ a - y
        return (K @ residual) / n + self.lmbda * (K @ a)

    def hessian(self, K):
        n = K.shape[0]
        return (K @ K) / n + self.lmbda * K

    def closed_form_solution(self, K, y):
        n = K.shape[0]
        mat = K + n * self.lmbda * np.eye(n)
        return np.linalg.solve(mat, y)

# -------------------------------------------------------------
# 3. LES 7 OPTIMISEURS
# -------------------------------------------------------------
class GD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, theta, grad):
        return theta - self.lr * grad

class GDLineSearch:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, theta, grad, hessian=None):
        if hessian is not None:
            g_flat = grad.ravel()
            gtg = np.dot(g_flat, g_flat)
            gHg = np.dot(g_flat, hessian @ g_flat)
            eta = (gtg / gHg) if gHg > 1e-12 else self.lr
            return theta - eta * grad
        return theta - self.lr * grad

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, theta, grad):
        return theta - self.lr * grad

class Momentum:
    def __init__(self, lr=0.01, mu=0.9):
        self.lr = lr
        self.mu = mu
        self.v = None
    def step(self, theta, grad):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.mu * self.v + grad
        return theta - self.lr * self.v

class AdaGrad:
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.s = None
    def step(self, theta, grad):
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s += grad ** 2
        return theta - self.lr * grad / (np.sqrt(self.s) + self.eps)

class RMSprop:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.s = None
    def step(self, theta, grad):
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s = self.rho * self.s + (1.0 - self.rho) * (grad ** 2)
        return theta - self.lr * grad / (np.sqrt(self.s) + self.eps)

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.s = None
        self.t = 0
    def step(self, theta, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.s = np.zeros_like(theta)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1.0 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        s_hat = self.s / (1.0 - self.beta2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(s_hat) + self.eps)

def get_optimizer(name, lr):
    if name == "GD": return GD(lr=lr)
    if name == "GDLineSearch": return GDLineSearch(lr=lr)
    if name == "SGD": return SGD(lr=lr)
    if name == "Momentum": return Momentum(lr=lr, mu=0.9)
    if name == "AdaGrad": return AdaGrad(lr=lr, eps=1e-8)
    if name == "RMSprop": return RMSprop(lr=lr, rho=0.9, eps=1e-8)
    if name == "Adam": return Adam(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)

# -------------------------------------------------------------
# 4. EXÉCUTION DU BENCHMARK
# -------------------------------------------------------------
def main():
    print("=================================================================")
    print("   LANCEMENT EXERCICE 1 SUR regression_nonlinear.csv")
    print("=================================================================")

    # Détection automatique du chemin du fichier
    if os.path.exists("data/regression_nonlinear.csv"):
        csv_path = "data/regression_nonlinear.csv"
    elif os.path.exists("regression_nonlinear.csv"):
        csv_path = "regression_nonlinear.csv"
    else:
        raise FileNotFoundError("regression_nonlinear.csv introuvable dans . ou data/")

    df = pd.read_csv(csv_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    feat_cols = ["x1", "x2"]
    X_tr_raw = train_df[feat_cols].values
    y_tr_raw = train_df["y"].values
    X_va_raw = val_df[feat_cols].values
    y_va_raw = val_df["y"].values
    X_te_raw = test_df[feat_cols].values
    y_te_raw = test_df["y"].values

    # Standardisation sur Train uniquement
    mean_X, std_X = compute_scaler(X_tr_raw)
    mean_y, std_y = compute_scaler(y_tr_raw)

    X_tr_s = apply_scaler(X_tr_raw, mean_X, std_X)
    X_va_s = apply_scaler(X_va_raw, mean_X, std_X)
    X_te_s = apply_scaler(X_te_raw, mean_X, std_X)

    y_tr_s = apply_scaler(y_tr_raw, mean_y, std_y)
    y_va_s = apply_scaler(y_va_raw, mean_y, std_y)
    y_te_s = apply_scaler(y_te_raw, mean_y, std_y)

    X_tr_lin = np.hstack([np.ones((X_tr_s.shape[0], 1)), X_tr_s])
    X_va_lin = np.hstack([np.ones((X_va_s.shape[0], 1)), X_va_s])
    X_te_lin = np.hstack([np.ones((X_te_s.shape[0], 1)), X_te_s])

    best_lmbda = 0.1
    best_k_lmb, best_k_sig = 1e-4, 0.5

    models_config = [
        ("OLS", LinearRegressionModel(lmbda=0.0), 0.0, None),
        ("Ridge", LinearRegressionModel(lmbda=best_lmbda), best_lmbda, None),
        ("KernelRidge", KernelRidgeRegressionModel(lmbda=best_k_lmb, sigma=best_k_sig), best_k_lmb, best_k_sig)
    ]

    opt_names = ["GD", "GDLineSearch", "SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"]
    lrs = {
        "GD": 0.05,
        "GDLineSearch": 0.05,
        "SGD": 0.02,
        "Momentum": 0.02,
        "AdaGrad": 0.2,
        "RMSprop": 0.01,
        "Adam": 0.05
    }

    seeds = [42, 123, 2024]
    batch_size = 32
    max_epochs = 100
    all_results = []

    for model_name, model_obj, lmb_val, sig_val in models_config:
        is_kernel = (model_name == "KernelRidge")
        if is_kernel:
            X_tr = rbf_kernel_matrix(X_tr_s, X_tr_s, model_obj.gamma)
            X_va = rbf_kernel_matrix(X_va_s, X_tr_s, model_obj.gamma)
            X_te = rbf_kernel_matrix(X_te_s, X_tr_s, model_obj.gamma)
            d_dim = X_tr.shape[0]
            H_exact = model_obj.hessian(X_tr)
        else:
            X_tr, X_va, X_te = X_tr_lin, X_va_lin, X_te_lin
            d_dim = X_tr.shape[1]
            H_exact = model_obj.hessian(X_tr)

        for opt_name in opt_names:
            seed_metrics = []
            for seed in seeds:
                np.random.seed(seed)
                theta = np.zeros(d_dim)
                opt = get_optimizer(opt_name, lr=lrs[opt_name])

                n_updates = 0
                n_grad_calls = 0
                best_epoch = 0
                best_v_rmse = float("inf")
                best_theta = theta.copy()
                t0 = time.time()
                n_samples = X_tr.shape[0]

                for epoch in range(1, max_epochs + 1):
                    if opt_name in ["SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"]:
                        indices = np.random.permutation(n_samples)
                        for b_start in range(0, n_samples, batch_size):
                            b_idx = indices[b_start:b_start + batch_size]
                            X_b, y_b = X_tr[b_idx], y_tr_s[b_idx]
                            if is_kernel:
                                res_b = X_b @ theta - y_b
                                grad = (X_b.T @ res_b) / len(b_idx) + model_obj.lmbda * (X_tr @ theta)
                            else:
                                res_b = X_b @ theta - y_b
                                P = model_obj._get_P(d_dim)
                                grad = (X_b.T @ res_b) / len(b_idx) + model_obj.lmbda * (P @ theta)
                            theta = opt.step(theta, grad)
                            n_updates += 1
                            n_grad_calls += 1
                    else:
                        grad = model_obj.gradient(theta, X_tr, y_tr_s)
                        n_grad_calls += 1
                        if opt_name == "GDLineSearch":
                            theta = opt.step(theta, grad, hessian=H_exact)
                        else:
                            theta = opt.step(theta, grad)
                        n_updates += 1

                    pred_val_s = X_va @ theta
                    pred_val_orig = invert_scaler(pred_val_s, mean_y, std_y)
                    _, v_rmse, _, _ = evaluate_metrics(y_va_raw, pred_val_orig)
                    if v_rmse < best_v_rmse:
                        best_v_rmse = v_rmse
                        best_epoch = epoch
                        best_theta = theta.copy()

                exec_time = time.time() - t0
                pred_te_s = X_te @ best_theta
                pred_te_orig = invert_scaler(pred_te_s, mean_y, std_y)
                te_mse, te_rmse, te_mae, te_r2 = evaluate_metrics(y_te_raw, pred_te_orig)

                seed_metrics.append({
                    "val_rmse": best_v_rmse,
                    "test_mse": te_mse,
                    "test_rmse": te_rmse,
                    "test_mae": te_mae,
                    "test_r2": te_r2,
                    "best_epoch": best_epoch,
                    "updates": n_updates,
                    "grad_calls": n_grad_calls,
                    "time": exec_time
                })

            all_results.append({
                "model": model_name,
                "optimizer": opt_name,
                "best_epoch": int(np.mean([m["best_epoch"] for m in seed_metrics])),
                "val_rmse": np.mean([m["val_rmse"] for m in seed_metrics]),
                "test_mse": np.mean([m["test_mse"] for m in seed_metrics]),
                "test_rmse": np.mean([m["test_rmse"] for m in seed_metrics]),
                "test_mae": np.mean([m["test_mae"] for m in seed_metrics]),
                "test_r2": np.mean([m["test_r2"] for m in seed_metrics]),
                "updates": int(np.mean([m["updates"] for m in seed_metrics])),
                "time_sec": np.mean([m["time"] for m in seed_metrics])
            })

    res_df = pd.DataFrame(all_results)
    res_df.to_csv("results/metrics_ex1_nonlinear.csv", index=False)
    print("\n=== RÉSULTATS DANS results/metrics_ex1_nonlinear.csv ===")
    print(res_df[["model", "optimizer", "val_rmse", "test_rmse", "test_r2", "updates", "time_sec"]].to_string(index=False))

    # Génération Figure
    km = KernelRidgeRegressionModel(lmbda=best_k_lmb, sigma=best_k_sig)
    K_tr = rbf_kernel_matrix(X_tr_s, X_tr_s, km.gamma)
    K_te = rbf_kernel_matrix(X_te_s, X_tr_s, km.gamma)
    a_star = km.closed_form_solution(K_tr, y_tr_s)
    pred_te_orig = invert_scaler(K_te @ a_star, mean_y, std_y)
    res_te = y_te_raw - pred_te_orig

    ols = LinearRegressionModel(lmbda=0.0)
    w_star = ols.closed_form_solution(X_tr_lin, y_tr_s)
    pred_ols_orig = invert_scaler(X_te_lin @ w_star, mean_y, std_y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(y_te_raw, pred_ols_orig, color='tab:blue', alpha=0.6, label='OLS (Lineaire)')
    axes[0].scatter(y_te_raw, pred_te_orig, color='tab:red', alpha=0.8, edgecolors='k', label='Kernel Ridge (RBF)')
    y_lims = [min(y_te_raw.min(), pred_te_orig.min()), max(y_te_raw.max(), pred_te_orig.max())]
    axes[0].plot(y_lims, y_lims, 'k--', lw=2, label='Ideal y = y_hat')
    axes[0].set_title("Predictions vs Vrai y (Test)")
    axes[0].set_xlabel("Vrai y")
    axes[0].set_ylabel("y_hat")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].hist(res_te, bins=15, color='tab:red', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_title("Residus (Kernel Ridge - Test)")
    axes[1].set_xlabel("Erreur (y - y_hat)")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    gx, gy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    grid_pts = np.c_[gx.ravel(), gy.ravel()]
    grid_pts_s = apply_scaler(grid_pts, mean_X, std_X)
    K_grid = rbf_kernel_matrix(grid_pts_s, X_tr_s, km.gamma)
    pred_grid = invert_scaler(K_grid @ a_star, mean_y, std_y).reshape(gx.shape)

    cnt = axes[2].contourf(gx, gy, pred_grid, levels=25, cmap='coolwarm', alpha=0.85)
    fig.colorbar(cnt, ax=axes[2], label='y_hat')
    axes[2].scatter(train_df['x1'], train_df['x2'], c='black', s=10, alpha=0.5, label='Train')
    axes[2].set_title("Surface Kernel Ridge (x1, x2)")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("figures/ex1_nonlinear_analysis.png", dpi=150)
    plt.close()
    print("Figure enregistree : figures/ex1_nonlinear_analysis.png")

if __name__ == "__main__":
    main()