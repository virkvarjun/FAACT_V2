"""Reversibility head: a small MLP regressing R(s) in [0, 1].

Deliberately the same shape as v1's failure predictor — `input_dim -> 256 -> 128 -> 1`
with ReLU and dropout — so that if this works and v1 did not, the difference is the
*target*, not the architecture. That is the claim under test: `failure_within_k` asks "am I
near the end of an episode that failed", which is proximity to termination; reversibility
asks "can I still come back from here".

Two deliberate choices:

- **Sigmoid output with soft-target BCE.** The labels are fractions (wins/M), not classes,
  and BCE against a soft target is the proper scoring rule for that — it is minimised
  exactly when the prediction equals the true probability. MSE would also fit, but is worse
  calibrated near 0 and 1, which is precisely where the horizon controller acts.
- **We report Spearman, not just loss.** The controller only needs the *ordering* of states
  to be right — it maps R to a horizon monotonically — so rank correlation is the metric
  that matches how the estimate is actually used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class ReversibilityHead(nn.Module):
    """MLP over ACT features, predicting reversibility in [0, 1]."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.1):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")

        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits; apply sigmoid for R. Logits keep BCE numerically stable."""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict R for a batch of standardised feature vectors."""
        self.eval()
        x = torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)
        return torch.sigmoid(self(x)).cpu().numpy()


class ScorerAdapter:
    """Adapts a trained head into the `predict_one` interface the controller expects.

    Bundles the standardiser with the model so the runtime cannot accidentally feed raw
    features to a model trained on standardised ones — a silent, catastrophic mismatch.
    """

    def __init__(self, head: ReversibilityHead, standardiser, feature_keys: list[str]) -> None:
        self.head = head
        self.standardiser = standardiser
        self.feature_keys = feature_keys

    def predict_one(self, features: dict[str, np.ndarray]) -> float:
        from faact.backbone.act_wrapper import concat_features

        x = concat_features(features, self.feature_keys)[None, :]
        return float(self.head.predict(self.standardiser.transform(x))[0])


@dataclass
class TrainResult:
    """Everything needed to report the M4 gate honestly."""

    epochs_run: int
    best_epoch: int
    train_loss: float
    val_loss: float
    seconds: float


def train_head(
    head: ReversibilityHead,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
    seed: int = 0,
    device: str = "cpu",
) -> TrainResult:
    """Train with soft-target BCE and early stopping on validation loss.

    Restores the best-validation weights at the end, so the reported test number comes
    from the checkpoint that was selected on validation — not from whatever the last epoch
    happened to leave behind.
    """
    import copy
    import time

    torch.manual_seed(seed)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()  # accepts soft targets in [0, 1]

    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    best_val, best_epoch, best_state, train_loss = float("inf"), 0, None, float("nan")
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(Xt.shape[0], generator=generator).to(device)
        epoch_loss = 0.0
        for i in range(0, Xt.shape[0], batch_size):
            idx = order[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(head(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item() * idx.numel()
        train_loss = epoch_loss / Xt.shape[0]

        head.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(head(Xv), yv))

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(head.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    return TrainResult(
        epochs_run=epoch,
        best_epoch=best_epoch,
        train_loss=train_loss,
        val_loss=best_val,
        seconds=time.perf_counter() - start,
    )


def evaluate(head: ReversibilityHead, X: np.ndarray, y: np.ndarray) -> dict:
    """Spearman / Pearson / MAE against measured R. Spearman is the M4 gate."""
    from scipy.stats import pearsonr, spearmanr

    pred = head.predict(X)
    # Correlation is undefined against a constant target, and would come back as nan
    # rather than as an error — report it explicitly instead.
    if np.std(y) < 1e-9 or np.std(pred) < 1e-9:
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "mae": float(np.mean(np.abs(pred - y))),
            "n": int(len(y)),
            "degenerate": True,
        }

    return {
        "spearman": float(spearmanr(pred, y).statistic),
        "pearson": float(pearsonr(pred, y).statistic),
        "mae": float(np.mean(np.abs(pred - y))),
        "n": int(len(y)),
        "degenerate": False,
    }
