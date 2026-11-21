"""Every predictor behind one interface:

    m = make("tabpfn")                 # or "cv", "cv_band", "lgbm"
    m.fit(X, Y)                        # Y: [n, 2] residual targets (ddx, ddy)
    Q = m.predict_quantiles(X, qs)     # [n, 2, len(qs)]

Residual targets mean constant velocity is the all-zeros prediction.
"""
from __future__ import annotations

import os
import time

import numpy as np

QS = (0.1, 0.5, 0.9)


class ConstantVelocity:
    """Zero residual, zero-width band."""
    name = "cv"

    def fit(self, X, Y):
        return self

    def predict_quantiles(self, X, qs=QS):
        return np.zeros((len(X), 2, len(qs)))


class ConstantBand:
    """Constant velocity median plus a fixed band: residual quantiles measured on the
    training rows, per horizon (first feature column). Answers 'why not a bigger margin?'."""
    name = "cv_band"

    def fit(self, X, Y):
        self.h = X[:, 0]; self.Y = Y
        return self

    def predict_quantiles(self, X, qs=QS):
        out = np.zeros((len(X), 2, len(qs)))
        for h in np.unique(X[:, 0]):
            m = np.isclose(self.h, h)
            ref = self.Y[m] if m.any() else self.Y
            q = np.quantile(ref, qs, axis=0).T          # [2, len(qs)]
            a = np.asarray(qs)                          # CV is the median: the band straddles 0
            q = np.where(a < 0.5, np.minimum(q, 0.0), np.where(a > 0.5, np.maximum(q, 0.0), 0.0))
            out[np.isclose(X[:, 0], h)] = q
        return out


class LGBMQuantile:
    name = "lgbm"

    def __init__(self, n_estimators=300, min_child_samples=10, num_leaves=31, n_jobs=None):
        self.n, self.mcs, self.leaves, self.n_jobs = n_estimators, min_child_samples, num_leaves, n_jobs

    def fit(self, X, Y):
        self.X, self.Y = X, Y
        self.models = {}
        return self

    def predict_quantiles(self, X, qs=QS):
        from lightgbm import LGBMRegressor
        out = np.zeros((len(X), 2, len(qs)))
        for ax in range(2):
            for j, q in enumerate(qs):
                key = (ax, q)
                if key not in self.models:
                    m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=self.n, learning_rate=0.05,
                                      num_leaves=self.leaves, min_child_samples=self.mcs, verbose=-1,
                                      **({"n_jobs": self.n_jobs} if self.n_jobs else {}))
                    self.models[key] = m.fit(self.X, self.Y[:, ax])
                out[:, ax, j] = self.models[key].predict(X)
        out.sort(axis=2)                                  # no crossing quantiles
        return out


def best_device():
    """cuda > mps (Apple GPU) > cpu, resolved here so the CPU large-context guard below still applies."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TabPFNQuantile:
    name = "tabpfn"

    def __init__(self, device=None, n_estimators=None, fast=False, fit_mode=None, api=False):
        import torch
        self.api = api                # hosted TabPFN (tabpfn-client): flat price per predict call
        self.device = "api" if api else (device or best_device())
        self.n_estimators = n_estimators
        self.fast = fast
        self.fit_mode = fit_mode      # "fit_with_cache" for many small predicts (closed loop)

    def _new(self):
        if self.api:
            import tabpfn_client
            from tabpfn_client import TabPFNRegressor as ClientRegressor
            tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
            return ClientRegressor(**({"n_estimators": self.n_estimators} if self.n_estimators else {}))
        from tabpfn import TabPFNRegressor
        kw = {"device": self.device}
        if self.n_estimators:
            kw["n_estimators"] = self.n_estimators
        if self.fit_mode:
            kw["fit_mode"] = self.fit_mode
        if self.fast:
            from tabpfn.constants import ModelVersion
            return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST, **kw)
        return TabPFNRegressor(**kw)

    def fit(self, X, Y):
        if self.device == "cpu" and len(X) > 5000:
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        t0 = time.perf_counter()
        self.models = [self._new().fit(X, Y[:, ax]) for ax in range(2)]
        self.t_fit = time.perf_counter() - t0
        return self

    def predict_quantiles(self, X, qs=QS):
        t0 = time.perf_counter()
        out = np.zeros((len(X), 2, len(qs)))
        for ax, m in enumerate(self.models):
            q = m.predict(X, output_type="quantiles", quantiles=list(qs))
            out[:, ax, :] = np.stack(q, axis=1)
        self.t_pred = time.perf_counter() - t0
        return out


def make(name, **kw):
    """name may carry LightGBM settings: "lgbm:m50:n100" = min_child_samples 50, 100 trees."""
    if name.startswith("lgbm:"):
        for part in name.split(":")[1:]:
            kw[{"m": "min_child_samples", "n": "n_estimators", "l": "num_leaves"}[part[0]]] = int(part[1:])
        name = "lgbm"
    return {"cv": ConstantVelocity, "cv_band": ConstantBand, "lgbm": LGBMQuantile,
            "tabpfn": TabPFNQuantile}[name](**kw)
