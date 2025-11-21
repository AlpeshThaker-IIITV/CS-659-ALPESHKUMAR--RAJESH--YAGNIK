# Week5_3.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
PROCESSED_DIR = "Processed"
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots")
TICKERS = ["AAPL", "TSLA", "MSFT"]
os.makedirs(PLOTS_DIR, exist_ok=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # input: df with Date, Return, State
    df = df.sort_values("Date").copy()
    df["Return_Sq"] = df["Return"] ** 2
    df["RollMean_10"] = df["Return"].rolling(window=10).mean()
    df["RollStd_10"] = df["Return"].rolling(window=10).std()
    df = df.dropna().reset_index(drop=True)
    return df


def cluster_and_save(ticker: str):
    path = os.path.join(PROCESSED_DIR, f"{ticker}_HMM_states.csv")
    if not os.path.exists(path):
        print(f"  Missing {path}, skipping.")
        return

    df = pd.read_csv(path, parse_dates=["Date"])
    df = create_features(df)
    features = df[["Return", "Return_Sq", "RollMean_10", "RollStd_10"]].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(Xs)
    df["Cluster"] = labels

    # PCA for visualization
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]

    out_csv = os.path.join(PROCESSED_DIR, f"{ticker}_clusters.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved clusters to {out_csv}")

    # Plot clusters
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=df, palette="tab10")
    plt.title(f"{ticker} — KMeans clusters (PCA view)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{ticker}_clusters.png"), dpi=300)
    plt.close()


def main():
    for t in TICKERS:
        print(f"\nClustering for {t} ...")
        cluster_and_save(t)
    print("\nDone: Week5_3 (clustering).")

main()
