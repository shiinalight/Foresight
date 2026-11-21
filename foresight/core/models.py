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
            q[:, list(qs).index(0.5)] = 0.0 if 0.5 in qs else q[:, 0]
            out[np.isclose(X[:, 0], h)] = q
        return out


class ConstantBox:
    """Constant velocity plus a fixed-size box centred on it, one box per horizon.

    Fit it on VALIDATION rows (never test): the box's along/across half-widths keep the shape of
    the residuals' per-axis 80th percentiles and are scaled so `target` of the validation
    positions fall inside the box (both axes at once). predict_quantiles returns -half, 0, +half
    whatever qs asks for: it is a box, not quantiles.
    """
    name = "cv_box"

    def __init__(self, target=0.8):
        self.target = target

    def fit(self, X, Y):
        self.half = {}
        for h in np.unique(X[:, 0]):
            Yh = np.abs(Y[np.isclose(X[:, 0], h)])
            shape = np.maximum(np.quantile(Yh, 0.8, axis=0), 1e-6)
            scale = np.quantile((Yh / shape).max(axis=1), self.target)
            self.half[float(h)] = shape * scale
        return self

    def predict_quantiles(self, X, qs=QS):
        out = np.zeros((len(X), 2, len(qs)))
        for h, half in self.half.items():
            m = np.isclose(X[:, 0], h)
            out[m, :, 0], out[m, :, -1] = -half, half
        return out


class LGBMQuantile:
    name = "lgbm"

    def __init__(self, n_estimators=300):
        self.n = n_estimators

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
                    m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=self.n,
                                      learning_rate=0.05, num_leaves=31, min_child_samples=10, verbose=-1)
                    self.models[key] = m.fit(self.X, self.Y[:, ax])
                out[:, ax, j] = self.models[key].predict(X)
        out.sort(axis=2)                                  # no crossing quantiles
        return out


class TabPFNQuantile:
    name = "tabpfn"

    def __init__(self, device=None, n_estimators=None, fast=False):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_estimators = n_estimators
        self.fast = fast

    def _new(self):
        from tabpfn import TabPFNRegressor
        kw = {"device": self.device}
        if self.n_estimators:
            kw["n_estimators"] = self.n_estimators
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


class TabPFNClientQuantile:
    """TabPFN through Prior Labs' hosted API (tabpfn_client): no GPU needed. Token from TABPFN_TOKEN.
    One regressor per axis; fit uploads the context, predict uses API quota per row."""
    name = "tabpfn_client"

    def fit(self, X, Y):
        from tabpfn_client import TabPFNRegressor, set_access_token
        token = os.environ.get("TABPFN_TOKEN")
        if not token:
            raise RuntimeError("TABPFN_TOKEN is not set: export TABPFN_TOKEN=<key from ux.priorlabs.ai/account>")
        set_access_token(token)
        t0 = time.perf_counter()
        self.models = [TabPFNRegressor().fit(X, Y[:, ax]) for ax in range(2)]
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
    return {"cv": ConstantVelocity, "cv_band": ConstantBand, "cv_box": ConstantBox, "lgbm": LGBMQuantile,
            "tabpfn": TabPFNQuantile, "tabpfn_client": TabPFNClientQuantile}[name](**kw)
