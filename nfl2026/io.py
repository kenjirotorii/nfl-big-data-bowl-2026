import pickle
from pathlib import Path
from typing import Any


def save_pickle(data: Any, filename: str | Path) -> None:
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_pickle(filename: str | Path) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)
