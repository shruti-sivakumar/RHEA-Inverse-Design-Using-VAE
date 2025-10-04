# src/train_cvae.py

import os, random
import numpy as np
import joblib, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from cvae import CVAE, kl_divergence

# -----------------------------
# Reproducibility
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_SPLITS = BASE_DIR / "data" / "processed" / "data_splits.npz"
MODEL_BEST = BASE_DIR / "models" / "cvae_best.pt"
MODEL_LAST = BASE_DIR / "models" / "cvae_last.pt"
MODEL_BEST.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Dataset wrapper
# -----------------------------
class TabDataset(Dataset):
    def __init__(self, X, y_cond, y_prop):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cond = torch.tensor(y_cond, dtype=torch.float32)
        self.y_prop = torch.tensor(y_prop, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y_cond[i], self.y_prop[i]

# -----------------------------
# Training functions
# -----------------------------
def train_epoch(model, loader, opt, device, epoch, warmup_epochs=50, beta_max=2.0, lambda_prop=1.0):
    model.train()
    beta = beta_max * min(1.0, (epoch + 1) / warmup_epochs)
    recon_losses, kl_losses, prop_losses = [], [], []
    for Xb, ycb, ypb in loader:
        Xb, ycb, ypb = Xb.to(device), ycb.to(device), ypb.to(device)
        X_hat, y_prop_pred, (q_mu, q_logvar, p_mu, p_logvar) = model(Xb, ycb)

        recon = F.mse_loss(X_hat, Xb)
        kl = kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
        prop = F.mse_loss(y_prop_pred, ypb)

        loss = recon + beta * kl + lambda_prop * prop
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        recon_losses.append(recon.item()); kl_losses.append(kl.item()); prop_losses.append(prop.item())
    return np.mean(recon_losses), np.mean(kl_losses), np.mean(prop_losses), beta

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    recon_losses, kl_losses, prop_losses, prop_maes = [], [], [], []
    for Xb, ycb, ypb in loader:
        Xb, ycb, ypb = Xb.to(device), ycb.to(device), ypb.to(device)
        X_hat, y_prop_pred, (q_mu, q_logvar, p_mu, p_logvar) = model(Xb, ycb)

        recon = F.mse_loss(X_hat, Xb)
        kl = kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
        prop = F.mse_loss(y_prop_pred, ypb)
        mae = F.l1_loss(y_prop_pred, ypb)

        recon_losses.append(recon.item()); kl_losses.append(kl.item())
        prop_losses.append(prop.item()); prop_maes.append(mae.item())
    return np.mean(recon_losses), np.mean(kl_losses), np.mean(prop_losses), np.mean(prop_maes)

# -----------------------------
# Main training loop
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load splits
    data = np.load(DATA_SPLITS)
    X_train, X_val = data["X_train"], data["X_val"]
    y_cond_train, y_cond_val = data["y_cond_train"], data["y_cond_val"]
    y_prop_train, y_prop_val = data["y_prop_train"], data["y_prop_val"]

    # DataLoaders
    train_loader = DataLoader(TabDataset(X_train, y_cond_train, y_prop_train),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TabDataset(X_val, y_cond_val, y_prop_val),
                            batch_size=64, shuffle=False)

    # Model
    model = CVAE(
        x_dim=X_train.shape[1],
        y_cond_dim=y_cond_train.shape[1],
        y_prop_dim=y_prop_train.shape[1],
        z_dim=8, hidden=256
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    best_val, patience, bad = 1e9, 30, 0
    for epoch in range(200):
        tr_recon, tr_kl, tr_prop, beta = train_epoch(model, train_loader, opt, device, epoch)
        va_recon, va_kl, va_prop, va_mae = eval_epoch(model, val_loader, device)
        val_score = va_recon + va_prop  # score: reconstruction + property loss

        print(f"Ep {epoch:03d} | β={beta:.2f} | "
              f"Train R {tr_recon:.4f} KL {tr_kl:.4f} P {tr_prop:.4f} | "
              f"Val R {va_recon:.4f} KL {va_kl:.4f} P {va_prop:.4f} MAE {va_mae:.4f}")

        # Save best model
        if val_score < best_val - 1e-5:
            best_val, bad = val_score, 0
            torch.save(model.state_dict(), MODEL_BEST)
        else:
            bad += 1

        if bad >= patience:
            print("Early stopping.")
            break

    # Save last model too
    torch.save(model.state_dict(), MODEL_LAST)
    print(f"[OK] Training finished. Best → {MODEL_BEST}, Last → {MODEL_LAST}")

if __name__ == "__main__":
    main()