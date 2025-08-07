from __future__ import annotations

import pickle
from pathlib import Path


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy-2-authored pickles from NumPy-1 runtimes."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as fp:
        return NumpyCompatUnpickler(fp).load()


def dump_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(obj, fp)
