import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class GaussianT3MF:
    """Third-order Gaussian membership with uncertain mean."""
    mean: float
    sigma: float
    eta: float
    samples: int = 5

    def membership(self, x: float) -> float:
        """Compute membership by sampling the uncertain mean."""
        if self.eta == 0:
            centers = np.array([self.mean])
        else:
            centers = np.random.normal(self.mean, self.eta, self.samples)
        return float(np.mean(np.exp(-((x - centers) ** 2) / (2 * self.sigma ** 2))))


class ANFIST3:
    """Minimal ANFIS of type-3 with Gaussian membership functions."""

    def __init__(self, n_inputs: int, mf_params: Sequence[Sequence[Tuple[float, float, float]]]):
        self.n_inputs = n_inputs
        self.mfs: List[List[GaussianT3MF]] = [[GaussianT3MF(*p) for p in mf_params[i]] for i in range(n_inputs)]
        self.rules = list(itertools.product(*[range(len(m)) for m in self.mfs]))
        self.consequents = np.zeros((len(self.rules), n_inputs + 1))

    def _firing_strengths(self, x: Sequence[float]) -> np.ndarray:
        mu = [[mf.membership(x[i]) for mf in self.mfs[i]] for i in range(self.n_inputs)]
        w = []
        for rule in self.rules:
            prod = 1.0
            for i, mf_idx in enumerate(rule):
                prod *= mu[i][mf_idx]
            w.append(prod)
        w = np.array(w, dtype=float)
        if np.sum(w) == 0:
            return np.ones_like(w) / len(w)
        return w / np.sum(w)

    def predict_row(self, x: Sequence[float]) -> float:
        w = self._firing_strengths(x)
        X = np.append(x, 1.0)
        y_rule = self.consequents @ X
        return float(np.dot(w, y_rule))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_row(x) for x in X])

    def train_ls(self, X: np.ndarray, y: np.ndarray) -> None:
        rows = []
        for x in X:
            w = self._firing_strengths(x)
            row = []
            for wi in w:
                row.extend(wi * np.append(x, 1.0))
            rows.append(row)
        A = np.array(rows)
        params, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.consequents = params.reshape(len(self.rules), self.n_inputs + 1)


def train_with_ex2() -> float:
    base = Path(__file__).resolve().parents[2]
    data = np.loadtxt(base / "Input" / "ex2.txt")
    X, y = data[:, :2], data[:, 2]
    mf_params = []
    for i in range(X.shape[1]):
        xmin, xmax = X[:, i].min(), X[:, i].max()
        sigma = (xmax - xmin) / 2 or 1.0
        eta = sigma / 4
        mf_params.append([(xmin, sigma, eta), (xmax, sigma, eta)])
    anfis = ANFIST3(n_inputs=2, mf_params=mf_params)
    anfis.train_ls(X, y)
    y_pred = anfis.predict(X)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    print(f"RMSE: {rmse:.4f}")
    for i, cons in enumerate(anfis.consequents):
        coeffs = ", ".join(f"{c:.4f}" for c in cons)
        print(f"Rule {i} consequents: {coeffs}")
    for i, mfs in enumerate(anfis.mfs):
        for j, mf in enumerate(mfs):
            print(
                f"Input {i} MF {j}: mean={mf.mean:.4f}, "
                f"sigma={mf.sigma:.4f}, eta={mf.eta:.4f}"
            )
    return rmse


if __name__ == "__main__":  # pragma: no cover
    train_with_ex2()
