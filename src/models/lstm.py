"""LSTM mu estimator — the most complex arm of the model-complexity ladder.

Where the linear and tree models consume hand-engineered tabular features,
the LSTM consumes a *sequence* of the last ``LSTM_SEQUENCE_LENGTH`` daily
log returns and learns its own internal representation. This is exactly
what LSTMs are designed for, and it lets the comparison cleanly answer:
"does adding recurrent dynamics buy us anything over flat-feature models?"

Training setup (matches GBT / linreg as closely as architecturally possible)
---------------------------------------------------------------------------
- Same target: cumulative log return over the next 21 trading days.
- Same global pooling: one model trained across all assets, with an
  embedding layer giving each asset its own learnable bias.
- Same backtest interface: returns a callable matching
  ``mu_estimator(window_returns, current_date) -> np.ndarray``.

A small, lightly-regularized LSTM is used to keep training fast on the
~3,000 daily observations available — overfit a giant network here and
the comparison becomes a story about regularization choices, not
architecture.
"""

import logging
import os
from typing import Callable

# Belt-and-braces OpenMP fix in case this module is imported directly without
# the entry-point script setting these. See run_complexity_ladder.py for the
# full explanation.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)

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

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


class _LSTMReturnForecaster(nn.Module):
    """LSTM with a learned per-asset embedding, predicting forward return."""

    def __init__(
        self,
        n_assets: int,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        embedding_dim: int = 4,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(n_assets, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        asset_ids: torch.Tensor,
    ) -> torch.Tensor:
        # sequences: (batch, seq_len, 1) — daily log returns
        # asset_ids: (batch,) — long
        _, (h_n, _) = self.lstm(sequences)
        last_hidden = h_n[-1]                        # (batch, hidden)
        embed = self.embedding(asset_ids)            # (batch, embed_dim)
        x = torch.cat([last_hidden, embed], dim=1)
        return self.head(x).squeeze(-1)              # (batch,)


def _build_sequences(
    log_returns: pd.DataFrame,
    sequence_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Construct training tensors from the wide log-returns DataFrame.

    For each (date t, asset i) where both the trailing window of length
    ``sequence_length`` and the forward window of length ``horizon`` are
    fully observed, emit one example: the trailing returns as input, the
    cumulative forward return as the target.

    Args:
        log_returns: DataFrame of daily log returns (T x N).
        sequence_length: Number of past days used as input.
        horizon: Number of forward days summed for the target.

    Returns:
        sequences (n_examples, sequence_length), asset_ids (n_examples,),
        targets (n_examples,), and the corresponding prediction dates.
    """
    arr = log_returns.values  # (T, N)
    n_rows, n_assets = arr.shape

    sequences: list[np.ndarray] = []
    asset_ids: list[int] = []
    targets: list[float] = []
    dates: list[pd.Timestamp] = []

    for t in range(sequence_length, n_rows - horizon):
        forward = arr[t + 1 : t + 1 + horizon]  # (horizon, N) — next H days
        forward_sum = forward.sum(axis=0)       # (N,)
        seq = arr[t - sequence_length : t]      # (seq_len, N)
        for i in range(n_assets):
            sequences.append(seq[:, i])
            asset_ids.append(i)
            targets.append(float(forward_sum[i]))
            dates.append(log_returns.index[t])

    return (
        np.asarray(sequences, dtype=np.float32)[:, :, None],  # (N, seq, 1)
        np.asarray(asset_ids, dtype=np.int64),
        np.asarray(targets, dtype=np.float32),
        dates,
    )


def train_lstm(
    log_returns: pd.DataFrame,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    horizon: int = GBT_FORECAST_HORIZON,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
    learning_rate: float = LSTM_LEARNING_RATE,
    patience: int = LSTM_PATIENCE,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[_LSTMReturnForecaster, list[str]]:
    """Train the LSTM forecaster on the wide log-returns DataFrame.

    Time-based hold-out for early stopping: the most recent ``val_fraction``
    of dates becomes the validation set, no shuffling.

    Args:
        log_returns: DataFrame of daily log returns (T x N).
        sequence_length: Past-days window used as input.
        horizon: Forecast horizon in trading days.
        epochs: Maximum training epochs (early stopping may end sooner).
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        patience: Early-stopping patience (epochs without val improvement).
        val_fraction: Fraction of most recent prediction-dates held out.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (fitted model, asset column ordering).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    asset_order = list(log_returns.columns)
    sequences, asset_ids, targets, dates = _build_sequences(
        log_returns, sequence_length, horizon
    )
    if len(sequences) == 0:
        raise ValueError("No training examples — series too short for given windows.")

    unique_dates = sorted(set(dates))
    cutoff_idx = max(1, int(len(unique_dates) * (1 - val_fraction)))
    cutoff_date = unique_dates[cutoff_idx]
    date_arr = np.asarray(dates)
    train_mask = date_arr < cutoff_date
    val_mask = ~train_mask

    def _make_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(sequences[mask]),
            torch.from_numpy(asset_ids[mask]),
            torch.from_numpy(targets[mask]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(train_mask, shuffle=True)
    val_loader = _make_loader(val_mask, shuffle=False)

    model = _LSTMReturnForecaster(n_assets=len(asset_order), sequence_length=sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    logger.info(
        "Training LSTM: train_examples=%d, val_examples=%d, n_assets=%d",
        int(train_mask.sum()), int(val_mask.sum()), len(asset_order),
    )

    # Track best val loss for logging only — by default we train for all
    # epochs because our natural val window overlaps the COVID crash, which
    # makes early stopping fire too aggressively.
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
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= max(1, int(train_mask.sum()))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_batch, id_batch, y_batch in val_loader:
                pred = model(seq_batch, id_batch)
                val_loss += loss_fn(pred, y_batch).item() * len(y_batch)
        val_loss /= max(1, int(val_mask.sum()))

        logger.info(
            "Epoch %d/%d  train=%.6f  val=%.6f", epoch, epochs, train_loss, val_loss
        )

        # Always remember the best-val state — restored at the end.
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
    return model, asset_order


def lstm_mu_estimator(
    model: _LSTMReturnForecaster,
    log_returns: pd.DataFrame,
    asset_order: list[str],
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    horizon: int = GBT_FORECAST_HORIZON,
) -> Callable[[pd.DataFrame, pd.Timestamp], np.ndarray]:
    """Build a mu-estimator callable for ``rolling_backtest``.

    Args:
        model: Fitted LSTM forecaster.
        log_returns: Full daily log-returns DataFrame (used to fetch the
            trailing sequence at each prediction date).
        asset_order: Canonical asset ordering returned to MVO.
        sequence_length: Trailing-days window the LSTM consumes.
        horizon: Forecast horizon (used to annualize predictions).

    Returns:
        Callable matching the rolling_backtest mu_estimator interface.
    """
    annualization = _TRADING_DAYS_PER_YEAR / float(horizon)
    asset_to_id = {a: i for i, a in enumerate(asset_order)}

    def estimator(
        window_returns: pd.DataFrame,  # noqa: ARG001 — interface symmetry
        current_date: pd.Timestamp,
    ) -> np.ndarray:
        # Get the trailing sequence_length rows up to (and including) current_date.
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

        seq_block = log_returns.iloc[t_idx - sequence_length : t_idx]  # (seq_len, N)
        seq_tensor = torch.from_numpy(
            seq_block[asset_order].values.astype(np.float32).T[:, :, None]
        )  # (n_assets, seq_len, 1)
        ids = torch.tensor([asset_to_id[a] for a in asset_order], dtype=torch.long)

        with torch.no_grad():
            preds = model(seq_tensor, ids).cpu().numpy()
        return preds.astype(float) * annualization

    return estimator
