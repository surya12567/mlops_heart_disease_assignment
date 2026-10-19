import pandas as pd
import pytest

from src.data import normalize_dataset, load_dataset


def test_normalize_dataset_converts_target_to_binary():
    x = pd.DataFrame({"Age": [50, 60, 70], "Chest Pain": [1, 2, 3]})
    y = pd.Series([0, 1, 4])
    frame = normalize_dataset(x, y)
    assert list(frame.columns) == ["age", "chest_pain", "target"]
    assert frame["target"].tolist() == [0, 1, 1]


def test_load_dataset_requires_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [50]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target"):
        load_dataset(path)


def test_load_dataset_rejects_empty(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"target": []}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path)
