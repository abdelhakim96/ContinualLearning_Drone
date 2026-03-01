"""
Pre-train the MLP Inverse Model for BlueROV2.

Trains the base MLP on the generated dataset and saves the model weights.
The saved model is later loaded by the DNN controller (frozen) and by
the LoRA continual-learning controllers (base frozen, adapters trainable).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bluerov.models.mlp import MLP

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__),
                          '..', 'data', 'dataset_bluerov_inverse.csv')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'base_inverse_mlp.pth')

# ── Hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN      = (256, 256, 128)
IN_DIM      = 9
OUT_DIM     = 4
EPOCHS      = 150
BATCH_SIZE  = 256
LR          = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT   = 0.15
SEED        = 42


def load_data(path: str):
    df   = pd.read_csv(path)
    feat_cols = ['e_surge', 'e_sway', 'e_heave', 'sin_epsi', 'cos_epsi',
                 'u', 'v', 'w', 'r']
    labl_cols = ['X_norm', 'Y_norm', 'Z_norm', 'Mz_norm']
    X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    Y = torch.tensor(df[labl_cols].values, dtype=torch.float32)
    return X, Y


def train(epochs=EPOCHS, lr=LR, hidden=HIDDEN, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run training/generate_dataset.py first.")

    X, Y = load_data(DATA_PATH)
    dataset = TensorDataset(X, Y)
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MLP(in_dim=IN_DIM, out_dim=OUT_DIM, hidden=hidden).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base MLP parameters: {total_params:,}")

    opt       = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= n_val

        scheduler.step()
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved → {MODEL_PATH}")
    return model, history


def load_trained_model(device: str = 'cpu') -> MLP:
    """Load the pre-trained base MLP."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run this script to train first.")
    model = MLP(in_dim=IN_DIM, out_dim=OUT_DIM, hidden=HIDDEN)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    return model


if __name__ == '__main__':
    train()
