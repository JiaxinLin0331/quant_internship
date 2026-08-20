from pathlib import Path
import pandas as pd


def load_ohlcv(data_dir):
    data_dir = Path(data_dir)

    csv_files = [
        file for file in data_dir.glob("*.csv")
        if file.name != "cleaned_market_data.csv"
    ]

    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)

        ticker = file.stem
        df["Ticker"] = ticker

        all_data.append(df)

    market_data = pd.concat(all_data, ignore_index=True)

    market_data["Date"] = pd.to_datetime(market_data["Date"])

    market_data = market_data.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
        "Ticker": "ticker"
    })

    market_data = market_data.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    return market_data



def validate_ohlcv(data):
    required_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume"
    ]

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close"
    ]

    # Missing values in original OHLCV columns only
    missing_values = data[required_columns].isnull().sum().sum()

    # Duplicate ticker-date observations
    duplicate_rows = data.duplicated(
        subset=["ticker", "date"]
    ).sum()

    # Prices must be positive
    invalid_prices = (
        data[price_columns] <= 0
    ).sum().sum()

    # Volume cannot be negative
    negative_volume = (
        data["volume"] < 0
    ).sum()

    # Basic OHLC consistency checks
    invalid_ohlc = (
        (data["high"] < data["low"]) |
        (data["high"] < data["open"]) |
        (data["high"] < data["close"]) |
        (data["low"] > data["open"]) |
        (data["low"] > data["close"])
    ).sum()

    return {
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "invalid_prices": invalid_prices,
        "negative_volume": negative_volume,
        "invalid_ohlc": invalid_ohlc
    }


import numpy as np

def compute_returns(data, frequency="daily"):
    df = data.copy()

    # Sort by ticker and date
    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    # Daily simple return
    df["simple_return"] = (
        df.groupby("ticker")["adjusted_close"]
        .pct_change()
    )

    # Daily log return
    df["log_return"] = (
        df.groupby("ticker")["adjusted_close"]
        .transform(lambda x: np.log(x / x.shift(1)))
    )

    # Return daily data
    if frequency == "daily":
        return df

    # Set resampling frequency
    if frequency == "weekly":
        freq = "W-FRI"
    elif frequency == "monthly":
        freq = "ME"
    else:
        raise ValueError(
            "frequency must be 'daily', 'weekly', or 'monthly'"
        )

    # Aggregate simple returns using compounding
    simple_returns = (
        df.dropna(subset=["simple_return"])
        .set_index("date")
        .groupby("ticker")["simple_return"]
        .resample(freq)
        .apply(lambda x: (1 + x).prod() - 1)
        .rename("simple_return")
    )

    # Aggregate log returns using summation
    log_returns = (
        df.dropna(subset=["log_return"])
        .set_index("date")
        .groupby("ticker")["log_return"]
        .resample(freq)
        .sum()
        .rename("log_return")
    )

    # Combine results
    returns = pd.concat(
        [simple_returns, log_returns],
        axis=1
    ).reset_index()

    return returns



def performance_metrics(data, risk_free_rate=0.04):
    # Annualized mean return
    annualized_return = (
        data.groupby("ticker")["simple_return"]
        .mean()
        * 252
    )

    # Annualized volatility
    annualized_volatility = (
        data.groupby("ticker")["simple_return"]
        .std()
        * np.sqrt(252)
    )

    # Sharpe ratio
    sharpe_ratio = (
        (annualized_return - risk_free_rate)
        / annualized_volatility
    )

    # Combine results
    metrics = pd.DataFrame({
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio
    })

    return metrics