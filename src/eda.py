"""Generate EDA summaries and visualizations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import DATA_PATH, load_dataset

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"


def run_eda(path: Path = DATA_PATH) -> dict:
    frame = load_dataset(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    x = frame.drop(columns="target")
    summary = {
        "rows": int(frame.shape[0]),
        "feature_count": int(x.shape[1]),
        "columns": list(frame.columns),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().items()},
        "target_counts": {str(k): int(v) for k, v in frame["target"].value_counts().sort_index().items()},
        "numeric_describe": frame.describe(include="all").fillna("").to_dict(),
    }
    (REPORT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    for col in x.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=frame, x=col, hue="target", kde=False, element="bars", stat="count")
        plt.title(f"Distribution of {col} by target")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"hist_{col}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=frame, x="target")
    plt.title("Heart disease target class balance (0=no disease, 1=disease)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(frame.corr(numeric_only=True), cmap="vlag", center=0, annot=False)
    plt.title("Feature and target correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    print(f"EDA complete. Figures: {FIG_DIR}")
    print(f"Rows={len(frame)}, features={x.shape[1]}")
    print("Missing values:", frame.isna().sum().to_dict())
    print("Target counts:", frame["target"].value_counts().to_dict())
    return summary


if __name__ == "__main__":
    run_eda()
