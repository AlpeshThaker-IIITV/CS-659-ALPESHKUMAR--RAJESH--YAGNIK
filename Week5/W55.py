#Week5_5.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mean_squared_error, r2_score

tickers = ["AAPL", "TSLA", "MSFT"]
data_dir = "Processed"
plot_dir = os.path.join(data_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Light theme (consistent with Week5_2.py)
sns.set_style("whitegrid")
plt.style.use("default")

def train_hmm_model(df, n_states=4):
    """Train an HMM on stock returns."""
    returns = df["Return"].dropna().values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )
    model.fit(returns)

    hidden_states = model.predict(returns)
    predicted_returns = model.means_[hidden_states].flatten()

    rmse = np.sqrt(mean_squared_error(returns, predicted_returns))
    r2 = r2_score(returns, predicted_returns)

    return model, rmse, r2

metrics = []

for ticker in tickers:
    csv_path = os.path.join(data_dir, f"{ticker}_HMM_states.csv")

    if not os.path.exists(csv_path):
        print(f"Missing file for {ticker}: {csv_path}")
        continue

    print(f"Processing {ticker} ...")
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Train HMM and get metrics
    best_model, rmse, r2 = train_hmm_model(df)
    metrics.append({"Ticker": ticker, "RMSE": rmse, "R2": r2})

    plt.figure(figsize=(6, 4))
    sns.heatmap(best_model.transmat_, annot=True, fmt=".3f", cmap="Blues")
    plt.title(f"HMM Transition Matrix – {ticker}")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{ticker}_transition_matrix.png"))
    plt.close()

if len(metrics) == 0:
    print("\n No valid ticker files found. Please ensure Week5_2.py outputs are in 'Processed/'.")
else:
    metrics_df = pd.DataFrame(metrics)
    print("\nModel Evaluation Summary:")
    print(metrics_df)


    plt.figure(figsize=(7, 5))
    sns.barplot(data=metrics_df, x="Ticker", y="RMSE", hue="Ticker", palette="coolwarm", legend=False)
    plt.title("HMM Model Performance (RMSE)")
    plt.xlabel("Ticker")
    plt.ylabel("Root Mean Squared Error")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "HMM_RMSE_Comparison.png"))
    plt.close()

    # R² Plot
    plt.figure(figsize=(7, 5))
    sns.barplot(data=metrics_df, x="Ticker", y="R2", hue="Ticker", palette="crest", legend=False)
    plt.title("HMM Model Performance (R² Score)")
    plt.xlabel("Ticker")
    plt.ylabel("R² Score")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "HMM_R2_Comparison.png"))
    plt.close()

    print("\nWeek5_5.py complete. All plots and metrics saved in Processed/plots/")
