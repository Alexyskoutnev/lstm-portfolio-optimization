"""Tests for the volatility regime module."""

import numpy as np
import pandas as pd

from src.data.regime import label_regimes_rolling_std, label_regimes_vix


class TestLabelRegimesVix:
    """Tests for VIX-based regime labeling."""

    def setup_method(self) -> None:
        """Create synthetic VIX data spanning all three regimes."""
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        values = np.concatenate([
            np.full(30, 12.0),   # Low
            np.full(30, 20.0),   # Medium
            np.full(30, 30.0),   # High
        ])
        self.vix = pd.Series(values, index=dates, name="VIX")

    def test_returns_series(self) -> None:
        """Should return a Series of regime labels."""
        labels = label_regimes_vix(self.vix)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(self.vix)

    def test_label_values(self) -> None:
        """Labels should be one of 'low', 'medium', 'high'."""
        labels = label_regimes_vix(self.vix)
        assert set(labels.unique()).issubset({"low", "medium", "high"})

    def test_low_regime_detected(self) -> None:
        """First 30 days (VIX=12) should be labeled 'low'."""
        labels = label_regimes_vix(self.vix, low_threshold=15.0, high_threshold=25.0)
        assert (labels.iloc[:30] == "low").all()

    def test_high_regime_detected(self) -> None:
        """Last 30 days (VIX=30) should be labeled 'high'."""
        labels = label_regimes_vix(self.vix, low_threshold=15.0, high_threshold=25.0)
        assert (labels.iloc[60:] == "high").all()

    def test_medium_regime_detected(self) -> None:
        """Middle 30 days (VIX=20) should be labeled 'medium'."""
        labels = label_regimes_vix(self.vix, low_threshold=15.0, high_threshold=25.0)
        assert (labels.iloc[30:60] == "medium").all()


class TestLabelRegimesRollingStd:
    """Tests for rolling-std-based regime labeling."""

    def setup_method(self) -> None:
        """Create synthetic returns with varying volatility."""
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        rng = np.random.default_rng(42)
        low_vol = rng.normal(0, 0.005, 150)
        high_vol = rng.normal(0, 0.03, 150)
        self.returns = pd.Series(
            np.concatenate([low_vol, high_vol]), index=dates, name="returns"
        )

    def test_returns_series(self) -> None:
        """Should return a Series of regime labels."""
        labels = label_regimes_rolling_std(self.returns, window=60)
        assert isinstance(labels, pd.Series)

    def test_no_nans_after_warmup(self) -> None:
        """After the rolling window warmup, no labels should be NaN."""
        labels = label_regimes_rolling_std(self.returns, window=60)
        assert labels.iloc[60:].isna().sum() == 0

    def test_label_values(self) -> None:
        """Labels should be one of 'low', 'medium', 'high', or NaN."""
        labels = label_regimes_rolling_std(self.returns, window=60)
        valid_labels = labels.dropna().unique()
        assert set(valid_labels).issubset({"low", "medium", "high"})
