"""
SynechismCore v19.0 — Training Utilities
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int = 64,
    name: str = "model",
    verbose: bool = True,
    device: torch.device = None,
) -> float:
    """Train a model and return best loss."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    dataset = TensorDataset(X_train, Y_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    model.train()

    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            # Handle shape: pred (B, steps, D), yb (B, steps, D)
            if pred.shape != yb.shape:
                min_steps = min(pred.shape[1], yb.shape[1])
                pred = pred[:, :min_steps, :]
                yb   = yb[:, :min_steps, :]
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        if avg < best_loss:
            best_loss = avg
        scheduler.step()

        if verbose and (epoch + 1) % 25 == 0:
            print(f"    [{name:>15}] epoch {epoch+1:>3}/{epochs} | loss={avg:.6f}")

    return best_loss


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    device: torch.device = None,
    batch_size: int = 128,
) -> tuple:
    """
    Evaluate model. Returns (predictions, true_values) as numpy arrays.
    Use stats.compute_full_stats() on these for proper significance testing.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i+batch_size].to(device)
            yb = Y_test[i:i+batch_size]
            pred = model(xb).cpu()
            min_steps = min(pred.shape[1], yb.shape[1])
            all_preds.append(pred[:, :min_steps, :].numpy())
            all_true.append(yb[:, :min_steps, :].numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_true, axis=0)
    return preds, trues
