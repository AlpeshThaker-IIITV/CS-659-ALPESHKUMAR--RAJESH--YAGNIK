# Week5_1.py
"""
Part 1: Download historical price data and save clean CSVs.
Saves files into Data/<TICKER>_full.csv with columns:
Date, Open, High, Low, Close, Adj Close, Volume
"""

import os
import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "TSLA", "MSFT"]
START_DATE = "2001-01-01"
END_DATE = "2025-10-28"
DATA_DIR = "Data"

os.makedirs(DATA_DIR, exist_ok=True)


def download_and_save(ticker: str):
    print(f"Downloading {ticker} {START_DATE} → {END_DATE} ...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=True, auto_adjust=False)

    if df.empty:
        print(f"  No data for {ticker}, skipping.")
        return

    # If columns are MultiIndex (ticker second-level), keep only first level names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Make Date a column (not the index)
    df = df.reset_index()

    # Drop rows that are entirely empty (except Date)
    df = df.dropna(how="all", subset=[c for c in df.columns if c != "Date"])

    # Standard column order if present
    wanted = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available = [c for c in wanted if c in df.columns]
    df = df[available]

    # Ensure Date column is datetime and remove invalid rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()

    out_path = os.path.join(DATA_DIR, f"{ticker}_full.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")


def main():
    for t in TICKERS:
        download_and_save(t)
    print("Done: Week5_1 (download & save).")


#if __name__ == "__main__":
main()
