from __future__ import annotations
import os
import math
import random
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore


class FallbackForecaster:
    """Lightweight forecaster using recent returns' EWMA and volatility.

    Produces mean, q10/q50/q90 and samples by assuming Gaussian with
    mean = ewma(returns) * horizon, std = vol15 * sqrt(horizon).
    """

    def __init__(self, samples: int = 100, horizon: int = 5) -> None:
        self.samples = samples
        self.horizon = horizon

    def _stats(self, X: np.ndarray) -> Tuple[float, float]:
        returns = X[:, 0]
        # EWMA with half-life ~10 steps
        alpha = 2.0 / (10.0 + 1.0)
        ewma = 0.0
        for r in returns[-60:]:
            ewma = alpha * float(r) + (1 - alpha) * ewma
        vol = float(np.std(returns[-15:])) if returns.size > 1 else 0.0
        mean = ewma * self.horizon
        std = max(1e-6, vol * math.sqrt(self.horizon))
        return mean, std

    def forecast(self, X: np.ndarray, last_price: float) -> Dict:
        mean, std = self._stats(X)
        samples = np.random.normal(loc=mean, scale=std, size=self.samples)
        # Build price path samples as relative cumulative returns
        cum = np.cumsum(samples.reshape(-1, 1) / max(1, self.horizon), axis=1)  # distribute over horizon
        # Generate simple linearly spaced steps for prices
        horizon_steps = self.horizon
        paths = []
        for s in samples:
            step = s / max(1, horizon_steps)
            prices = [last_price]
            price = last_price
            for _ in range(horizon_steps):
                price = price * math.exp(step)
                prices.append(price)
            paths.append(prices)
        q10 = float(np.quantile(samples, 0.10))
        q50 = float(np.quantile(samples, 0.50))
        q90 = float(np.quantile(samples, 0.90))
        return {
            "return_mean": float(mean),
            "return_q10": q10,
            "return_q50": q50,
            "return_q90": q90,
            "samples": paths,
        }


class RNNSampler(nn.Module if nn is not None else object):
    """Optional LSTM + linear head for mean/std.

    If torch is not installed, this class is a no-op placeholder.
    """

    def __init__(self, input_size: int, hidden_size: int = 96, layers: int = 2, dropout: float = 0.1, horizon: int = 5):
        if nn is None:
            return
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)  # mean, log_std for next-aggregate return

    def forward(self, x):  # type: ignore
        # x: (B, T, F)
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        out = self.head(last)
        mean, log_std = out[:, 0], out[:, 1]
        std = torch.nn.functional.softplus(log_std) + 1e-6
        return mean, std


class Forecaster:
    def __init__(self, window: int = 60, horizon: int = 5, samples: int = 100, artifact_path: Optional[str] = None) -> None:
        self.window = window
        self.horizon = horizon
        self.samples = samples
        self.artifact_path = artifact_path or os.environ.get("RNN_ARTIFACT", "services/model/artifacts/rnn_meanstd.pt")
        self.rnn: Optional[RNNSampler] = None
        if torch is not None:
            try:
                state = None
                if os.path.exists(self.artifact_path):
                    state = torch.load(self.artifact_path, map_location="cpu")
                if state is not None:
                    input_size = state.get("input_size", 6)
                    model = RNNSampler(input_size=input_size, hidden_size=state.get("hidden_size", 96), layers=state.get("layers", 2), dropout=state.get("dropout", 0.1), horizon=self.horizon)
                    model.load_state_dict(state["state_dict"])  # type: ignore
                    model.eval()
                    self.rnn = model
            except Exception:
                self.rnn = None
        self.fallback = FallbackForecaster(samples=samples, horizon=horizon)

    def forecast(self, X: np.ndarray, last_price: float) -> Dict:
        if self.rnn is None or torch is None:
            return self.fallback.forecast(X, last_price)
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).unsqueeze(0)  # (1, T, F)
            mean, std = self.rnn(x)
            mean = float(mean.item()) * self.horizon
            std = float(std.item()) * math.sqrt(self.horizon)
        samples = np.random.normal(loc=mean, scale=max(1e-6, std), size=self.samples)
        paths = []
        for s in samples:
            step = s / max(1, self.horizon)
            price = last_price
            points = [last_price]
            for _ in range(self.horizon):
                price = price * math.exp(step)
                points.append(price)
            paths.append(points)
        q10 = float(np.quantile(samples, 0.10))
        q50 = float(np.quantile(samples, 0.50))
        q90 = float(np.quantile(samples, 0.90))
        return {
            "return_mean": float(mean),
            "return_q10": q10,
            "return_q50": q50,
            "return_q90": q90,
            "samples": paths,
        }
