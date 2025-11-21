# Week5_2.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore", message="Model is not converging")
import logging
logging.getLogger("hmmlearn.hmm").setLevel(logging.ERROR)

sns.set_style("whitegrid")

DATA_DIR = "Data"
PROCESSED_DIR = "Processed"
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots")
TICKERS = ["AAPL", "TSLA", "MSFT"]
N_STATES_TRY = [2, 3, 4]
COV_TYPE = "full"
N_ITER = 200
RND = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def log_returns(series: pd.Series) -> pd.Series:

    return np.log(series / series.shift(1)).dropna()


def num_hmm_params(n_components: int, n_features: int, cov_type: str = "full") -> int:

    tm = n_components * (n_components - 1)  # transitions without last column constraint
    ip = (n_components - 1)  # initial probabilities
    means = n_components * n_features
    if cov_type == "full":
        covs = n_components * (n_features * (n_features + 1) // 2)
    else:
        covs = n_components * n_features
    return tm + ip + means + covs


def aic_bic(loglik: float, n_params: int, n_samples: int):
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n_samples) - 2 * loglik
    return aic, bic


def fit_best_hmm(returns_values: np.ndarray):

    models = {}
    results = []
    n_samples = returns_values.shape[0]

    for k in N_STATES_TRY:
        m = GaussianHMM(n_components=k, covariance_type=COV_TYPE, n_iter=N_ITER, random_state=RND)
        m.fit(returns_values)
        ll = m.score(returns_values)
        params = num_hmm_params(k, returns_values.shape[1], COV_TYPE)
        aic, bic = aic_bic(ll, params, n_samples)
        models[k] = m
        results.append({"states": k, "loglike": ll, "aic": aic, "bic": bic})
    res_df = pd.DataFrame(results).sort_values("bic").reset_index(drop=True)
    best_k = int(res_df.loc[0, "states"])
    return models[best_k], res_df


def save_plots_and_csv(ticker: str, df_prices: pd.DataFrame, df_states: pd.DataFrame, model, res_df):

    out_states = os.path.join(PROCESSED_DIR, f"{ticker}_HMM_states.csv")
    out_summary = os.path.join(PROCESSED_DIR, f"{ticker}_HMM_summary.csv")
    out_metrics = os.path.join(PROCESSED_DIR, f"{ticker}_HMM_metrics.csv")

    df_states.to_csv(out_states, index=False)

    summaries = []
    for s in sorted(df_states["State"].unique()):
        subset = df_states[df_states["State"] == s]
        summaries.append({
            "State": int(s),
            "Mean Return": float(subset["Return"].mean()),
            "Std Dev": float(subset["Return"].std()),
            "Occupancy (%)": float(100 * len(subset) / len(df_states))
        })
    pd.DataFrame(summaries).to_csv(out_summary, index=False)
    res_df.to_csv(out_metrics, index=False)


    fig, ax = plt.subplots(figsize=(11, 5))
    for s in sorted(df_states["State"].unique()):
        sel = df_states[df_states["State"] == s]
        ax.scatter(sel["Date"], sel["Return"], s=8, label=f"State {s}", alpha=0.7)
    ax.set_title(f"{ticker} — Log Returns by HMM State")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{ticker}_returns_states.png"), dpi=300)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df_prices["Date"], df_prices["Adj Close"], lw=1.25, label="Adj Close")
    # shade segments by decoded state (df_states is aligned to returns dates)
    states = df_states["State"].values
    dates = pd.to_datetime(df_states["Date"]).values
    start = 0
    cur_state = states[0]
    for i in range(1, len(states)):
        if states[i] != cur_state:
            ax.axvspan(dates[start], dates[i - 1], color=f"C{cur_state}", alpha=0.12)
            cur_state = states[i]
            start = i
    ax.axvspan(dates[start], dates[-1], color=f"C{cur_state}", alpha=0.12)
    ax.set_title(f"{ticker} — Price with HMM Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{ticker}_price_regimes.png"), dpi=300)
    plt.close(fig)


def process_ticker(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker}_full.csv")
    if not os.path.exists(path):
        print(f"  Missing {path}, skipping.")
        return

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "Adj Close" not in df.columns:
        print("  Adj Close missing, skipping.")
        return

    returns = log_returns(df["Adj Close"])
    if returns.empty:
        print("  Not enough data to compute returns, skipping.")
        return

    ret_vals = returns.values.reshape(-1, 1)
    model, res_df = fit_best_hmm(ret_vals)
    hidden = model.predict(ret_vals)


    df_states = pd.DataFrame({
        "Date": df.loc[returns.index, "Date"].values,
        "Return": returns.values,
        "State": hidden
    })
    # Save outputs and plots
    save_plots_and_csv(ticker, df, df_states, model, res_df)
    print(f"  Done: {ticker} (best states: {model.n_components})")


def main():
    for t in TICKERS:
        print(f"\nProcessing {t} ...")
        process_ticker(t)
    print("\nDone: Week5_2 (HMM processing).")

main()
