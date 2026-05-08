"""LSTM mu estimator — the most complex arm of the model-complexity ladder.

Where the linear and tree models consume hand-engineered tabular features,
the LSTM consumes a *sequence* of recent observations and learns its own
internal representation. This is exactly what LSTMs are designed for, and
it lets the comparison cleanly answer: "does adding recurrent dynamics buy
us anything over flat-feature models?"

What this module does carefully — and *why*
-------------------------------------------
A naive LSTM on raw daily returns hits a wall fast:

- Daily returns are tiny (≈ 0.01) so gradients are tiny — training stalls.
- A single channel (own return) is too sparse a signal — features beat
  flat-sequence models in equity prediction.
- 21-day-forward targets are noisy enough that MSE rewards "predict zero"
  more than it punishes it, so over-regularized models flat-line.

We address each of those:

1. **Multi-channel sequence input.** Every timestep gets multiple channels
   — own return, rolling vol, VIX level, market mean — instead of just
   the daily return. LSTM gets to model the joint dynamics.
2. **Per-channel standardization** using training-set stats only. Inputs
   become z-scored, gradients are well-scaled, training is stable.
3. **Standardized target.** The 21-day forward return is z-scored using
   training mean/std, predictions are un-standardized at inference time.
4. **Asset embedding** for the global model — gives each ETF a learned
   per-asset bias.
5. **Huber loss** — quadratic on small errors, linear on large ones.
   Robust to the fat-tailed return distribution (and to COVID-period
   outliers that dominate plain MSE).
6. **Larger hidden state and dropout** — needed to fit the richer input.

Training mirrors GBT/Ridge as closely as architecture allows: same target,
same global pooling, same backtest interface (a ``mu_estimator`` callable).
"""

import logging
import os
from collections.abc import Callable

# OpenMP env vars must be set before any torch import. See
# scripts/run_complexity_ladder.py for full explanation.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    GBT_FORECAST_HORIZON,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LEARNING_RATE,
    LSTM_NUM_LAYERS,
    LSTM_PATIENCE,
    LSTM_SEQUENCE_LENGTH,
)

torch.set_num_threads(1)
logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


class _LSTMReturnForecaster(nn.Module):
    """LSTM with a per-asset embedding, predicting standardized forward return."""

    def __init__(
        self,
        n_assets: int,
        n_input_channels: int,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(n_assets, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=n_input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        asset_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, (h_n, _) = self.lstm(sequences)
        last_hidden = h_n[-1]
        embed = self.embedding(asset_ids)
        x = torch.cat([last_hidden, embed], dim=1)
        return self.head(x).squeeze(-1)


def _build_channel_arrays(
    log_returns: pd.DataFrame,
    vix: pd.Series,
) -> dict[str, np.ndarray]:
    """Build per-asset channel arrays of shape (T, N).

    Each channel is one of the time series the LSTM sees at every step.
    All values at row t are observed at-or-before t-1 — strict no-look-ahead.

    Channels:
      - return       : daily log return at t (i.e. value of returns at row t)
      - vol_21       : trailing-21d realized vol per asset
      - vix          : VIX level (broadcast across assets)
      - market_mean  : same-day average return across assets (broadcast)
    """
    n_dates, n_assets = log_returns.shape
    out: dict[str, np.ndarray] = {}
    out["return"] = log_returns.values.astype(np.float32)
    out["vol_21"] = (
        log_returns.rolling(window=21).std().fillna(0.0).values.astype(np.float32)
    )
    vix_aligned = vix.reindex(log_returns.index).ffill().bfill().values.astype(np.float32)
    out["vix"] = np.broadcast_to(vix_aligned[:, None], (n_dates, n_assets)).copy()
    market_mean = log_returns.mean(axis=1).values.astype(np.float32)
    out["market_mean"] = np.broadcast_to(
        market_mean[:, None], (n_dates, n_assets)
    ).copy()
    return out


def _build_sequences(
    channels: dict[str, np.ndarray],
    sequence_length: int,
    horizon: int,
    train_end_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[str, tuple[float, float]]]:
    """Build a stacked tensor of (n_examples, sequence_length, n_channels).

    Standardizes each channel using the *training* statistics only. If
    ``train_end_idx`` is None, all examples count as training.

    Returns:
        sequences (n_examples, sequence_length, n_channels), asset_ids
        (n_examples,), targets (n_examples,), date row-indices, and a
        dict of per-channel (mean, std) used for standardization.
    """
    channel_names = list(channels.keys())
    arr = np.stack([channels[k] for k in channel_names], axis=-1)  # (T, N, C)
    n_dates, n_assets, n_channels = arr.shape

    # Compute training-only stats for each channel from the full panel.
    if train_end_idx is None:
        train_end_idx = n_dates
    stats: dict[str, tuple[float, float]] = {}
    for c, name in enumerate(channel_names):
        vals = arr[:train_end_idx, :, c]
        mu = float(np.nanmean(vals))
        sigma = float(np.nanstd(vals)) or 1.0
        stats[name] = (mu, sigma)
        arr[:, :, c] = (arr[:, :, c] - mu) / sigma

    sequences: list[np.ndarray] = []
    asset_ids: list[int] = []
    targets: list[float] = []
    date_idx: list[int] = []

    # Target uses *unstandardized* daily returns
    raw_returns = channels["return"]

    for t in range(sequence_length, n_dates - horizon):
        seq = arr[t - sequence_length : t]   # (seq_len, N, C)
        forward = raw_returns[t + 1 : t + 1 + horizon]  # (horizon, N)
        forward_sum = forward.sum(axis=0)
        for i in range(n_assets):
            sequences.append(seq[:, i, :])
            asset_ids.append(i)
            targets.append(float(forward_sum[i]))
            date_idx.append(t)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(asset_ids, dtype=np.int64),
        np.asarray(targets, dtype=np.float32),
        date_idx,
        stats,
    )


def train_lstm(
    log_returns: pd.DataFrame,
    vix: pd.Series,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    horizon: int = GBT_FORECAST_HORIZON,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
    learning_rate: float = LSTM_LEARNING_RATE,
    patience: int = LSTM_PATIENCE,
    val_fraction: float = 0.2,
    seed: int = 0,
    huber_delta: float = 0.05,
) -> tuple[_LSTMReturnForecaster, list[str], dict[str, tuple[float, float]], tuple[float, float]]:
    """Train the multi-channel LSTM forecaster.

    Args:
        log_returns: DataFrame of daily log returns (T x N).
        vix: Series of daily VIX values aligned to ``log_returns.index``.
        sequence_length: Past-days window used as input.
        horizon: Forecast horizon in trading days.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate (with cosine decay).
        patience: Early-stopping patience. Set very high to disable
            (default config does this — see why in the module docstring).
        val_fraction: Fraction of most-recent prediction-dates held out
            for the val loss only (no early stopping by default).
        seed: Random seed for reproducibility.
        huber_delta: Threshold for Huber loss.

    Returns:
        Tuple of:
        - fitted LSTM model
        - asset column ordering
        - per-channel input standardization stats {channel: (mu, sigma)}
        - target standardization stats (mu, sigma)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    asset_order = list(log_returns.columns)
    channels = _build_channel_arrays(log_returns, vix)

    # Build sequences using all data (train + val combined). Standardization
    # uses the whole training-data block as a stats source.
    sequences, asset_ids, targets, date_idx, channel_stats = _build_sequences(
        channels, sequence_length, horizon
    )
    if len(sequences) == 0:
        raise ValueError("No training examples — series too short for given windows.")

    # Time-based train/val split on the prediction-date index
    unique_dates = sorted(set(date_idx))
    cutoff_idx = max(1, int(len(unique_dates) * (1 - val_fraction)))
    cutoff_t = unique_dates[cutoff_idx]
    date_arr = np.asarray(date_idx)
    train_mask = date_arr < cutoff_t
    val_mask = ~train_mask

    # Standardize the target using *training* stats
    y_mu = float(targets[train_mask].mean())
    y_sigma = float(targets[train_mask].std()) or 1.0
    targets_z = (targets - y_mu) / y_sigma

    def _make_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(sequences[mask]),
            torch.from_numpy(asset_ids[mask]),
            torch.from_numpy(targets_z[mask].astype(np.float32)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(train_mask, shuffle=True)
    val_loader = _make_loader(val_mask, shuffle=False)

    n_channels = sequences.shape[-1]
    model = _LSTMReturnForecaster(
        n_assets=len(asset_order),
        n_input_channels=n_channels,
        sequence_length=sequence_length,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    logger.info(
        "Training LSTM: train=%d, val=%d, n_assets=%d, channels=%d (huber, target z-scored)",
        int(train_mask.sum()), int(val_mask.sum()), len(asset_order), n_channels,
    )

    use_early_stopping = patience > 0 and patience < epochs
    best_val = float("inf")
    epochs_without_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seq_batch, id_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(seq_batch, id_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= max(1, int(train_mask.sum()))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_batch, id_batch, y_batch in val_loader:
                pred = model(seq_batch, id_batch)
                val_loss += loss_fn(pred, y_batch).item() * len(y_batch)
        val_loss /= max(1, int(val_mask.sum()))

        logger.info(
            "Epoch %d/%d  train=%.6f  val=%.6f  lr=%.2e",
            epoch, epochs, train_loss, val_loss, scheduler.get_last_lr()[0],
        )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            epochs_without_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improve += 1
            if use_early_stopping and epochs_without_improve >= patience:
                logger.info("Early stopping at epoch %d (best val=%.6f)", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, asset_order, channel_stats, (y_mu, y_sigma)


def lstm_mu_estimator(
    model: _LSTMReturnForecaster,
    log_returns: pd.DataFrame,
    vix: pd.Series,
    asset_order: list[str],
    channel_stats: dict[str, tuple[float, float]],
    target_stats: tuple[float, float],
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    horizon: int = GBT_FORECAST_HORIZON,
) -> Callable[[pd.DataFrame, pd.Timestamp], np.ndarray]:
    """Build a mu-estimator callable for ``rolling_backtest``.

    Replicates the train-time channel construction and standardization
    exactly: build (return, vol, vix, market_mean) channels for the
    trailing window, z-score using training stats, predict, un-standardize
    the target, and annualize.

    Args:
        model: Fitted LSTM forecaster.
        log_returns: Full daily log-returns DataFrame.
        vix: Full VIX series aligned to log_returns.
        asset_order: Canonical asset ordering returned to MVO.
        channel_stats: Per-channel (mu, sigma) from training.
        target_stats: Target (mu, sigma) from training, used to invert
            the z-score at prediction time.
        sequence_length: Trailing-days window the LSTM consumes.
        horizon: Forecast horizon (used to annualize predictions).

    Returns:
        Callable matching the rolling_backtest mu_estimator interface.
    """
    annualization = _TRADING_DAYS_PER_YEAR / float(horizon)
    asset_to_id = {a: i for i, a in enumerate(asset_order)}
    y_mu, y_sigma = target_stats

    # Pre-compute the channels once over the full date range — re-doing this
    # at every rebalance call would be wasteful.
    channels = _build_channel_arrays(log_returns, vix)
    channel_names = list(channels.keys())
    # Stack and standardize using the cached training stats.
    arr_full = np.stack([channels[k] for k in channel_names], axis=-1)  # (T, N, C)
    for c, name in enumerate(channel_names):
        mu, sigma = channel_stats[name]
        arr_full[:, :, c] = (arr_full[:, :, c] - mu) / sigma

    def estimator(
        window_returns: pd.DataFrame,  # noqa: ARG001 — interface symmetry
        current_date: pd.Timestamp,
    ) -> np.ndarray:
        try:
            t_idx = log_returns.index.get_loc(current_date)
        except KeyError:
            logger.warning(
                "LSTM estimator: %s not in returns index; falling back to zero mu",
                current_date,
            )
            return np.zeros(len(asset_order))
        if not isinstance(t_idx, int):
            t_idx = int(np.asarray(t_idx).ravel()[0])
        if t_idx < sequence_length:
            return np.zeros(len(asset_order))

        seq_block = arr_full[t_idx - sequence_length : t_idx]  # (seq, N, C)
        # Reorder to (n_assets, seq, C) and predict for all assets at once
        seq_per_asset = np.transpose(seq_block, (1, 0, 2))
        seq_tensor = torch.from_numpy(seq_per_asset.astype(np.float32))
        ids = torch.tensor([asset_to_id[a] for a in asset_order], dtype=torch.long)

        with torch.no_grad():
            preds_z = model(seq_tensor, ids).cpu().numpy()
        # Un-standardize: predicted z * train_sigma + train_mu = predicted raw
        preds_raw = preds_z * y_sigma + y_mu
        return preds_raw.astype(float) * annualization

    return estimator
