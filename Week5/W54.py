# Week5_4.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

processed_dir = "Processed"
plots_dir = os.path.join(processed_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

tickers = ["AAPL", "TSLA", "MSFT"]

# helper: compute posterior by simple empirical approach:
# posterior ~ frequency of state at the last observed time (we can also use Viterbi counts).
# We also use empirical transition matrix to get next-step probabilities.
def empirical_transition(states):
    states = np.asarray(states, dtype=int)
    n = states.max() + 1
    counts = np.zeros((n, n), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = counts / row_sums
    trans = np.nan_to_num(trans)
    return trans

for ticker in tickers:
    states_file = os.path.join(processed_dir, f"{ticker}_HMM_states.csv")
    if not os.path.exists(states_file):
        print(f"Skipping {ticker}: {states_file} not found.")
        continue

    print(f"\nForecasting for {ticker} ...")
    df = pd.read_csv(states_file, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    if df.empty:
        print("  No state data, skipping.")
        continue

    # empirical transition and last-state
    trans = empirical_transition(df["State"].astype(int).values)
    last_state = int(df["State"].iloc[-1])

    # compute next-state probabilities
    if last_state < trans.shape[0]:
        next_prob = trans[last_state]  # row for last_state
    else:
        # fallback: uniform if mismatch
        next_prob = np.ones(trans.shape[0]) / trans.shape[0]

    # compute mean returns per state (empirical)
    state_means = df.groupby("State")["Return"].mean().to_dict()
    n_states = trans.shape[0]
    means = np.array([state_means.get(s, 0.0) for s in range(n_states)])

    expected_next_return = np.dot(next_prob, means)

    # save prediction details
    out_pred = {
        "Ticker": ticker,
        "LastDate": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "LastState": int(last_state),
        "ExpectedNextReturn": float(expected_next_return)
    }
    out_df = pd.DataFrame([out_pred])
    pred_csv = os.path.join(processed_dir, f"{ticker}_next_step_prediction.csv")
    out_df.to_csv(pred_csv, index=False)
    print(f"  Saved next-step prediction -> {pred_csv}")

    # Plot: next-state probability bar chart
    plt.figure(figsize=(6, 4))
    sns.barplot(x=np.arange(len(next_prob)), y=next_prob, palette="pastel")
    plt.title(f"{ticker} — Next-state probabilities (from state {last_state})")
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.tight_layout()
    prob_file = os.path.join(plots_dir, f"{ticker}_next_state_probs.png")
    plt.savefig(prob_file, dpi=300)
    plt.close()
    print(f"  Saved next-state probs plot -> {prob_file}")

    # Plot: expected-return bar (single value with context of state means)
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(n_states), means, color=plt.cm.Paired.colors[:n_states], alpha=0.8)
    plt.axhline(expected_next_return, color="k", linestyle="--", label="Expected next return")
    plt.title(f"{ticker} — State mean returns and expected next return")
    plt.xlabel("State")
    plt.ylabel("Mean return")
    plt.legend()
    plt.tight_layout()
    exp_file = os.path.join(plots_dir, f"{ticker}_expected_next_return.png")
    plt.savefig(exp_file, dpi=300)
    plt.close()
    print(f"  Saved expected-return plot -> {exp_file}")

print("\nWeek5_4 done: 1-step forecasts and plots saved.")
