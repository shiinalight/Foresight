"""MPS-vs-CPU parity for TabPFN quantiles (Nooshin's Mac). Same context rows, same queries, both devices.

    python tools/check_mps.py
Prints the largest absolute difference between the two devices' quantiles (metres) and the error and
joint 5-95 coverage of each. Small differences (float32 kernels) are expected; a difference that
changes coverage or error by more than ~0.01 means MPS results must not be mixed with CPU results.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.05, 0.1, 0.5, 0.9, 0.95)


def main():
    import torch
    if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        print("MPS not available on this machine: skipped"); return
    df = rows_for("data/crowd", 0.1, horizons=(1.0, 2.0))
    tr, _, te = F.split_by_episode(df)
    ctx, q = df[tr].sample(1000, random_state=0), df[te].sample(300, random_state=0)
    Y, out = q[["ddx", "ddy"]].to_numpy(), {}
    for dev in ("cpu", "mps"):
        m = M.make("tabpfn", device=dev, n_estimators=1).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
        Q = m.predict_quantiles(F.X_of(q), QS)
        err = np.linalg.norm(Q[:, :, 2] - Y, axis=1).mean()
        joint = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 4])).all(axis=1).mean()
        out[dev] = Q
        print(f"{dev}: err {err:.4f} m  joint 5-95 {100 * joint:.1f}%  (fit {m.t_fit:.2f} s, predict {m.t_pred:.2f} s)")
    print(f"largest |cpu - mps| over all quantiles: {np.abs(out['cpu'] - out['mps']).max():.4f} m")


if __name__ == "__main__":
    main()
