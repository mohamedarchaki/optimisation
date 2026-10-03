"""
Fichier : run_exercice2.py
Description : Exercice 2 - Classification Multiclasse (MLP 3 couches: 2 -> 32 -> 16 -> 3)
sur classification_spiral3.csv en pur NumPy.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 1. Standardisation calculée sur train uniquement
def compute_scaler(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std

def apply_scaler(arr, mean, std):
    return (arr - mean) / std

def one_hot(y, num_classes=3):
    n = len(y)
    Y = np.zeros((n, num_classes))
    Y[np.arange(n), y.astype(int)] = 1.0
    return Y

# 2. Réseau de neurones à 3 couches : tanh, tanh, softmax
class MLP3Layer:
    def __init__(self, d_in=2, d_h1=32, d_h2=16, d_out=3, lmbda=1e-4):
        self.d_in = d_in
        self.d_h1 = d_h1
        self.d_h2 = d_h2
        self.d_out = d_out
        self.lmbda = lmbda

    def init_params(self, seed=42):
        np.random.seed(seed)
        # Initialisation de type Xavier / Glorot
        scale1 = np.sqrt(2.0 / (self.d_in + self.d_h1))
        scale2 = np.sqrt(2.0 / (self.d_h1 + self.d_h2))
        scale3 = np.sqrt(2.0 / (self.d_h2 + self.d_out))

        W1 = np.random.randn(self.d_in, self.d_h1) * scale1
        b1 = np.zeros((1, self.d_h1))
        W2 = np.random.randn(self.d_h1, self.d_h2) * scale2
        b2 = np.zeros((1, self.d_h2))
        W3 = np.random.randn(self.d_h2, self.d_out) * scale3
        b3 = np.zeros((1, self.d_out))

        return [W1, b1, W2, b2, W3, b3]

    def forward(self, X, params):
        W1, b1, W2, b2, W3, b3 = params
        Z1 = X @ W1 + b1
        H1 = np.tanh(Z1)
        Z2 = H1 @ W2 + b2
        H2 = np.tanh(Z2)
        Z3 = H2 @ W3 + b3
        
        # Softmax stable (soustraction du maximum par ligne)
        shift_Z = Z3 - np.max(Z3, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        cache = (X, H1, H2, P)
        return P, cache

    def loss(self, params, X, Y_onehot):
        P, _ = self.forward(X, params)
        eps = 1e-15
        P_safe = np.clip(P, eps, 1.0 - eps)
        b_size = X.shape[0]
        ce_loss = -np.sum(Y_onehot * np.log(P_safe)) / b_size

        # Régularisation L2 uniquement sur les poids W_l
        W1, _, W2, _, W3, _ = params
        reg_loss = 0.5 * self.lmbda * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        return ce_loss + reg_loss

    def backward(self, params, cache, Y_onehot):
        W1, b1, W2, b2, W3, b3 = params
        X, H1, H2, P = cache
        b_size = X.shape[0]

        # Rétropropagation vectorisée
        D3 = (P - Y_onehot) / b_size
        dW3 = H2.T @ D3 + self.lmbda * W3
        db3 = np.sum(D3, axis=0, keepdims=True)

        D2 = (D3 @ W3.T) * (1.0 - H2 ** 2)
        dW2 = H1.T @ D2 + self.lmbda * W2
        db2 = np.sum(D2, axis=0, keepdims=True)

        D1 = (D2 @ W2.T) * (1.0 - H1 ** 2)
        dW1 = X.T @ D1 + self.lmbda * W1
        db1 = np.sum(D1, axis=0, keepdims=True)

        return [dW1, db1, dW2, db2, dW3, db3]

# 3. Les 7 Optimiseurs adaptés aux listes de paramètres
class BaseOpt:
    def __init__(self, lr=0.05):
        self.lr = lr
        self.t = 0
    def reset(self):
        self.t = 0

class GD(BaseOpt):
    def step(self, params, grads):
        self.t += 1
        return [p - self.lr * g for p, g in zip(params, grads)]

class GDLineSearch(BaseOpt):
    def step(self, params, grads, obj_func=None):
        self.t += 1
        eta = self.lr
        if obj_func is not None:
            c = 1e-4
            rho = 0.5
            curr_loss = obj_func(params)
            grad_norm_sq = sum(np.sum(g ** 2) for g in grads)
            for _ in range(10):
                trial_p = [p - eta * g for p, g in zip(params, grads)]
                if obj_func(trial_p) <= curr_loss - c * eta * grad_norm_sq:
                    break
                eta *= rho
        return [p - eta * g for p, g in zip(params, grads)]

class SGD(BaseOpt):
    def step(self, params, grads):
        self.t += 1
        return [p - self.lr * g for p, g in zip(params, grads)]

class Momentum(BaseOpt):
    def __init__(self, lr=0.05, mu=0.9):
        super().__init__(lr=lr)
        self.mu = mu
        self.v = None
    def reset(self):
        super().reset()
        self.v = None
    def step(self, params, grads):
        self.t += 1
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        self.v = [self.mu * vi + g for vi, g in zip(self.v, grads)]
        return [p - self.lr * vi for p, vi in zip(params, self.v)]

class AdaGrad(BaseOpt):
    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr=lr)
        self.eps = eps
        self.s = None
    def reset(self):
        super().reset()
        self.s = None
    def step(self, params, grads):
        self.t += 1
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]
        self.s = [si + g ** 2 for si, g in zip(self.s, grads)]
        return [p - self.lr * g / (np.sqrt(si) + self.eps) for p, g, si in zip(params, grads, self.s)]

class RMSprop(BaseOpt):
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        super().__init__(lr=lr)
        self.rho = rho
        self.eps = eps
        self.s = None
    def reset(self):
        super().reset()
        self.s = None
    def step(self, params, grads):
        self.t += 1
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]
        self.s = [self.rho * si + (1.0 - self.rho) * (g ** 2) for si, g in zip(self.s, grads)]
        return [p - self.lr * g / (np.sqrt(si) + self.eps) for p, g, si in zip(params, grads, self.s)]

class Adam(BaseOpt):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
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
    def step(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.s = [np.zeros_like(p) for p in params]
        self.m = [self.beta1 * mi + (1.0 - self.beta1) * g for mi, g in zip(self.m, grads)]
        self.s = [self.beta2 * si + (1.0 - self.beta2) * (g ** 2) for si, g in zip(self.s, grads)]
        m_hat = [mi / (1.0 - self.beta1 ** self.t) for mi in self.m]
        s_hat = [si / (1.0 - self.beta2 ** self.t) for si in self.s]
        return [p - self.lr * mhi / (np.sqrt(shi) + self.eps) for p, mhi, shi in zip(params, m_hat, s_hat)]

def evaluate_classifier(mlp, params, X, y_true):
    P, _ = mlp.forward(X, params)
    preds = np.argmax(P, axis=1)
    acc = np.mean(preds == y_true)

    classes = np.unique(y_true)
    f1s, precs, recs = [], [], []
    for c in classes:
        tp = np.sum((preds == c) & (y_true == c))
        fp = np.sum((preds == c) & (y_true != c))
        fn = np.sum((preds != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    return acc, np.mean(precs), np.mean(recs), np.mean(f1s), preds

def main():
    print("=================================================================")
    print("   LANCEMENT EXERCICE 2 SUR classification_spiral3.csv")
    print("=================================================================")

    csv_path = "data/classification_spiral3.csv" if os.path.exists("data/classification_spiral3.csv") else "classification_spiral3.csv"
    df = pd.read_csv(csv_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    feat_cols = ["x1", "x2"]
    X_tr_raw = train_df[feat_cols].values
    y_tr = train_df["label"].values.astype(int)
    X_va_raw = val_df[feat_cols].values
    y_va = val_df["label"].values.astype(int)
    X_te_raw = test_df[feat_cols].values
    y_te = test_df["label"].values.astype(int)

    mean_X, std_X = compute_scaler(X_tr_raw)
    X_tr = apply_scaler(X_tr_raw, mean_X, std_X)
    X_va = apply_scaler(X_va_raw, mean_X, std_X)
    X_te = apply_scaler(X_te_raw, mean_X, std_X)

    Y_tr_oh = one_hot(y_tr, 3)
    Y_va_oh = one_hot(y_va, 3)
    Y_te_oh = one_hot(y_te, 3)

    mlp = MLP3Layer(d_in=2, d_h1=32, d_h2=16, d_out=3, lmbda=1e-4)

    opt_configs = {
        "GD": 0.5,
        "GDLineSearch": 0.5,
        "SGD": 0.1,
        "Momentum": 0.08,
        "AdaGrad": 0.3,
        "RMSprop": 0.01,
        "Adam": 0.02
    }

    seeds = [42, 123, 2024]
    max_epochs = 150
    batch_size = 32
    results = []
    best_overall_params = None
    best_overall_acc = -1

    for opt_name, lr_val in opt_configs.items():
        seed_runs = []
        for seed in seeds:
            params = mlp.init_params(seed=seed)
            if opt_name == "GD": opt = GD(lr=lr_val)
            elif opt_name == "GDLineSearch": opt = GDLineSearch(lr=lr_val)
            elif opt_name == "SGD": opt = SGD(lr=lr_val)
            elif opt_name == "Momentum": opt = Momentum(lr=lr_val, mu=0.9)
            elif opt_name == "AdaGrad": opt = AdaGrad(lr=lr_val)
            elif opt_name == "RMSprop": opt = RMSprop(lr=lr_val)
            elif opt_name == "Adam": opt = Adam(lr=lr_val)

            n_updates = 0
            n_grad_calls = 0
            best_epoch = 0
            best_val_loss = float("inf")
            best_params = [p.copy() for p in params]
            t0 = time.time()
            n_samples = X_tr.shape[0]

            for epoch in range(1, max_epochs + 1):
                if opt_name in ["SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"]:
                    indices = np.random.permutation(n_samples)
                    for b_start in range(0, n_samples, batch_size):
                        b_idx = indices[b_start:b_start + batch_size]
                        X_b, Y_b = X_tr[b_idx], Y_tr_oh[b_idx]
                        _, cache = mlp.forward(X_b, params)
                        grads = mlp.backward(params, cache, Y_b)
                        params = opt.step(params, grads)
                        n_updates += 1
                        n_grad_calls += 1
                else:
                    _, cache = mlp.forward(X_tr, params)
                    grads = mlp.backward(params, cache, Y_tr_oh)
                    n_grad_calls += 1
                    if opt_name == "GDLineSearch":
                        obj_f = lambda p: mlp.loss(p, X_tr, Y_tr_oh)
                        params = opt.step(params, grads, obj_func=obj_f)
                    else:
                        params = opt.step(params, grads)
                    n_updates += 1

                # Validation sur perte
                v_loss = mlp.loss(params, X_va, Y_va_oh)
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    best_params = [p.copy() for p in params]

            exec_time = time.time() - t0
            acc_te, prec_te, rec_te, f1_te, preds_te = evaluate_classifier(mlp, best_params, X_te, y_te)
            ce_te = mlp.loss(best_params, X_te, Y_te_oh)

            seed_runs.append({
                "val_loss": best_val_loss,
                "test_acc": acc_te,
                "test_f1": f1_te,
                "test_ce": ce_te,
                "best_epoch": best_epoch,
                "updates": n_updates,
                "grad_calls": n_grad_calls,
                "time": exec_time
            })

            if acc_te > best_overall_acc:
                best_overall_acc = acc_te
                best_overall_params = [p.copy() for p in best_params]

        results.append({
            "optimizer": opt_name,
            "lr": lr_val,
            "best_epoch": int(np.mean([s["best_epoch"] for s in seed_runs])),
            "val_loss": np.mean([s["val_loss"] for s in seed_runs]),
            "test_acc": np.mean([s["test_acc"] for s in seed_runs]),
            "test_f1": np.mean([s["test_f1"] for s in seed_runs]),
            "test_ce": np.mean([s["test_ce"] for s in seed_runs]),
            "updates": int(np.mean([s["updates"] for s in seed_runs])),
            "time_sec": np.mean([s["time"] for s in seed_runs])
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv("results/metrics_ex2_spiral.csv", index=False)
    print("\n=== RÉSULTATS DANS results/metrics_ex2_spiral.csv ===")
    print(res_df[["optimizer", "lr", "best_epoch", "test_acc", "test_f1", "test_ce", "updates", "time_sec"]].to_string(index=False))

    # Figure : Frontières de Décision 2D + Matrice de Confusion 3x3
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_min, x_max = df['x1'].min() - 0.2, df['x1'].max() + 0.2
    y_min, y_max = df['x2'].min() - 0.2, df['x2'].max() + 0.2
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_raw = np.c_[gx.ravel(), gy.ravel()]
    grid_s = apply_scaler(grid_raw, mean_X, std_X)
    P_grid, _ = mlp.forward(grid_s, best_overall_params)
    Z_grid = np.argmax(P_grid, axis=1).reshape(gx.shape)

    axes[0].contourf(gx, gy, Z_grid, alpha=0.35, cmap='tab10')
    axes[0].scatter(df['x1'], df['x2'], c=df['label'], cmap='tab10', edgecolors='k', s=35)
    axes[0].set_title(f"Frontières de Décision MLP (Best Test Acc: {best_overall_acc*100:.1f}%)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    _, _, _, _, preds_te = evaluate_classifier(mlp, best_overall_params, X_te, y_te)
    conf_mat = np.zeros((3, 3), dtype=int)
    for t_val, p_val in zip(y_te, preds_te):
        conf_mat[t_val, p_val] += 1

    im = axes[1].imshow(conf_mat, cmap='Blues', interpolation='nearest')
    axes[1].set_title("Matrice de Confusion (Test 3x3)")
    axes[1].set_xlabel("Classe Prédite")
    axes[1].set_ylabel("Classe Réelle")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(conf_mat[i, j]), ha="center", va="center",
                         color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig("figures/ex2_spiral_analysis.png", dpi=150)
    plt.close()
    print("Figure enregistrée : figures/ex2_spiral_analysis.png")

if __name__ == "__main__":
    main()