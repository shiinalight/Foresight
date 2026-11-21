"""Check TabPFN access: the cloud client (API) and the local model. Never prints the key.

Run in the terminal where TABPFN_TOKEN is set:
  C:\\Users\\samco\\tools\\lerobot-test\\.venv-x64\\Scripts\\python.exe check_tabpfn.py
"""
import os
import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split

if not os.environ.get("TABPFN_TOKEN"):
    sys.exit("TABPFN_TOKEN is not set in this terminal. PowerShell: $env:TABPFN_TOKEN = '<key>'  (or setx TABPFN_TOKEN \"<key>\" and open a new terminal)")
print(f"TABPFN_TOKEN found (length {len(os.environ['TABPFN_TOKEN'])})")

Xc, yc = load_breast_cancer(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3, random_state=0)
Xr, yr = make_regression(n_samples=600, n_features=10, noise=5.0, random_state=0)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=0)


def report(name, fn):
    t0 = time.perf_counter()
    try:
        msg = fn()
        print(f"[OK]   {name}: {msg} ({time.perf_counter() - t0:.1f} s)")
    except Exception as e:  # report and carry on to the next check
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")


# ---- 1. cloud client (tabpfn-client) -------------------------------------
def client_cls():
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    m = TabPFNClassifier()
    m.fit(Xc_tr, yc_tr)
    return f"accuracy {(m.predict(Xc_te) == yc_te).mean():.3f} on {len(yc_te)} rows"


def client_reg():
    from tabpfn_client import TabPFNRegressor
    m = TabPFNRegressor()
    m.fit(Xr_tr, yr_tr)
    p = m.predict(Xr_te)
    return f"R2 {1 - ((p - yr_te) ** 2).sum() / ((yr_te - yr_te.mean()) ** 2).sum():.3f}"


report("client classifier", client_cls)
report("client regressor", client_reg)


# ---- 2. local model (tabpfn, CPU) ----------------------------------------
def local_cls():
    from tabpfn import TabPFNClassifier
    m = TabPFNClassifier(device="cpu")
    m.fit(Xc_tr, yc_tr)
    return f"accuracy {(m.predict(Xc_te) == yc_te).mean():.3f} on {len(yc_te)} rows"


def local_reg_timing():
    from tabpfn import TabPFNRegressor
    out = []
    for n in (200, 420):
        m = TabPFNRegressor(device="cpu")
        t0 = time.perf_counter(); m.fit(Xr_tr[:n], yr_tr[:n]); tf = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict(Xr_te); tp = time.perf_counter() - t0
        r2 = 1 - ((p - yr_te) ** 2).sum() / ((yr_te - yr_te.mean()) ** 2).sum()
        out.append(f"context {n}: R2 {r2:.3f}, fit {tf:.1f} s, predict {tp:.1f} s for {len(yr_te)} rows")
    return " | ".join(out)


report("local classifier (downloads weights on first run)", local_cls)
report("local regressor + timing", local_reg_timing)
