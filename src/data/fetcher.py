"""Fetch and cache financial price data from Yahoo Finance."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _build_cache_path(cache_dir: str, tickers: list[str], start: str, end: str) -> Path:
    """Build a deterministic cache file path for a given query.

    Args:
        cache_dir: Directory where cache files are stored.
        tickers: Ticker symbols included in the query.
        start: Start date string.
        end: End date string.

    Returns:
        Path to the CSV cache file.
    """
    filename = f"prices_{'_'.join(tickers)}_{start}_{end}.csv"
    return Path(cache_dir) / filename


def _read_cache(cache_path: Path) -> pd.DataFrame | None:
    """Return cached DataFrame if the file exists, otherwise ``None``.

    Args:
        cache_path: Path to the CSV cache file.

    Returns:
        Cached DataFrame or None on cache miss.
    """
    if cache_path.exists():
        logger.info("Cache hit: %s", cache_path)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    logger.info("Cache miss: %s", cache_path)
    return None


def _write_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Persist a DataFrame to CSV, creating parent directories as needed.

    Args:
        df: DataFrame to write.
        cache_path: Destination path.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    logger.debug("Wrote cache file: %s", cache_path)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_raw(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download raw OHLCV data from Yahoo Finance.

    Args:
        tickers: Ticker symbols to download.
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.

    Returns:
        Raw DataFrame returned by ``yf.download``.

    Raises:
        ValueError: If Yahoo Finance returns no data at all.
    """
    logger.info("Downloading data for %s from %s to %s", tickers, start, end)
    # group_by="ticker" ensures multi-ticker downloads produce a MultiIndex
    # with (ticker, field) columns, giving us a uniform structure to parse.
    raw: pd.DataFrame = yf.download(  # type: ignore[assignment]
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if raw.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for tickers {tickers} "
            f"in range {start} to {end}. Verify the symbols and date range."
        )
    return raw


def _extract_close_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract the Close column for each ticker from raw download data.

    Handles the structural difference between single-ticker downloads
    (which may return a flat column index) and multi-ticker downloads
    (which always return a ``pd.MultiIndex``).

    Args:
        raw: Raw DataFrame from ``yf.download``.
        tickers: Ticker symbols that were requested.

    Returns:
        DataFrame with one Close-price column per ticker.

    Raises:
        ValueError: If any requested ticker is missing from the result.
    """
    if len(tickers) == 1:
        # Single-ticker downloads may or may not have a MultiIndex depending
        # on the yfinance version; handle both layouts defensively.
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[(tickers[0], "Close")].to_frame(name=tickers[0])
        else:
            df = raw[["Close"]].rename(columns={"Close": tickers[0]})  # type: ignore[arg-type]
    else:
        df = pd.DataFrame(
            {
                ticker: raw[(ticker, "Close")]
                for ticker in tickers
                if (ticker, "Close") in raw
            },
            index=raw.index,
        )

    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(
            f"No price data returned for tickers: {missing}. "
            "They may be delisted or misspelled."
        )
    logger.debug("Extracted close prices for %d ticker(s)", len(tickers))
    return df


def _fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then back-fill missing prices.

    Different ETFs/stocks may observe different market holidays, leaving
    sporadic NaNs.  Forward-fill propagates the last known price into gaps,
    and back-fill covers any leading NaNs at the start of the series.

    Args:
        df: DataFrame that may contain NaN values.

    Returns:
        DataFrame with no missing values.

    Raises:
        ValueError: If gaps remain after filling (should not happen in
            practice but guards against fully-empty columns).
    """
    df = df.ffill().bfill()
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"Unable to fill all missing values in price data. "
            f"Columns still containing NaNs: {nan_cols}"
        )
    logger.debug("Filled missing prices; result shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers.

    Args:
        tickers: List of ticker symbols (e.g., ["XLK", "XLF"]).
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.
        cache_dir: Optional directory to cache downloaded data as CSV.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker,
        containing adjusted close prices with no missing values.

    Raises:
        ValueError: If no data is found for any of the requested tickers.
    """
    # --- Check cache ---
    cache_path: Path | None = None
    if cache_dir:
        cache_path = _build_cache_path(cache_dir, tickers, start, end)
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    # --- Download, extract, clean ---
    raw = _download_raw(tickers, start, end)
    df = _extract_close_prices(raw, tickers)
    df = _fill_missing_prices(df)

    # --- Persist to cache ---
    if cache_path is not None:
        _write_cache(df, cache_path)

    return df


def fetch_vix(start: str, end: str) -> pd.Series:
    """Download VIX close prices.

    Args:
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.

    Returns:
        Series with DatetimeIndex containing VIX close values, named "VIX".

    Raises:
        ValueError: If no VIX data is found.
    """
    logger.info("Downloading VIX data from %s to %s", start, end)
    raw: pd.DataFrame = yf.download(  # type: ignore[assignment]
        "^VIX", start=start, end=end, auto_adjust=True, progress=False
    )

    if raw.empty:
        raise ValueError(
            f"No VIX data found for date range {start} to {end}. "
            "The Yahoo Finance symbol ^VIX may be temporarily unavailable."
        )

    # yfinance may return a MultiIndex with ("Close", "^VIX") or a flat
    # "Close" column depending on the version; handle both.
    if isinstance(raw.columns, pd.MultiIndex):
        vix: pd.Series = raw[("Close", "^VIX")].squeeze()  # type: ignore[assignment]
    else:
        vix = raw["Close"].squeeze()  # type: ignore[assignment]

    vix.name = "VIX"

    # Forward-fill then back-fill to cover holiday gaps and leading NaNs.
    vix = vix.ffill().bfill()
    logger.debug("VIX data fetched; %d observations", len(vix))
    return vix  # type: ignore[return-value]
