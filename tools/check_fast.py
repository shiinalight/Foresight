"""F3: is TabPFN-3.5-Fast with 1 estimator as good as the default model across scenes?

Same 1,500 context rows and the same test rows for both, ETH->Hotel and Hotel->ETH, at +2 s:
error, per-axis coverage of 10-90%, joint coverage of the 5-95% box, width, pinball.

    python tools/check_fast.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import QS, scores  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

HZ = (1.2, 2.0, 4.8)
for src, dst in (("eth", "hotel"), ("hotel", "eth")):
    tr = rows_for(f"data/{src}", 0.0, horizons=HZ); te = rows_for(f"data/{dst}", 0.0, horizons=HZ)
    tr, te = tr[np.isclose(tr.h, 2.0)], te[np.isclose(te.h, 2.0)]
    ctx, t = tr.sample(1500, random_state=0), te.sample(min(600, len(te)), random_state=0)
    for name, kw in (("default", {}), ("fast, 1 est", {"fast": True, "n_estimators": 1})):
        t0 = time.perf_counter()
        m = M.make("tabpfn", **kw).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
        s = scores(m.predict_quantiles(F.X_of(t), QS), t[["ddx", "ddy"]].to_numpy())
        print(f"{src}->{dst} {name:<12} err {s['err_mean']:.3f}  per-axis80 {100 * s['cov80']:.0f}%  "
              f"joint box90 {100 * s['joint90']:.0f}%  width80 {s['width80']:.2f}  pinball {s['pinball']:.4f}  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
