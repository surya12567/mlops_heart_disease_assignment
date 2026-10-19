"""Download and prepare the UCI Heart Disease dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "heart.csv"


def normalize_dataset(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Create a clean table and convert the original 0–4 target to binary 0/1."""
    x = features.copy()
    x.columns = [str(c).strip().lower().replace(" ", "_") for c in x.columns]
    y = pd.to_numeric(target, errors="coerce")
    # In the UCI Cleveland data, 0 means no disease; 1–4 indicate disease.
    y = (y > 0).astype("Int64")
    frame = x.apply(pd.to_numeric, errors="coerce")
    frame["target"] = y
    frame = frame.dropna(subset=["target"])
    frame["target"] = frame["target"].astype(int)
    return frame.reset_index(drop=True)


def download_dataset(output_path: Path = DATA_PATH) -> Path:
    """Fetch dataset ID 45 from UCI via ucimlrepo and save it as CSV."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)
    frame = normalize_dataset(dataset.data.features, dataset.data.targets.iloc[:, 0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} rows and {len(frame.columns) - 1} features to {output_path}")
    print("Target distribution:")
    print(frame["target"].value_counts().sort_index().to_string())
    return output_path


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load a dataset CSV and validate the required target column."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m src.data --download` first."
        )
    frame = pd.read_csv(path)
    if "target" not in frame.columns:
        raise ValueError("Dataset must contain a 'target' column.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download dataset from UCI")
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    if args.download:
        download_dataset(args.output)
    else:
        frame = load_dataset(args.output)
        print(frame.head())
        print(f"\nShape: {frame.shape}")


if __name__ == "__main__":
    main()
