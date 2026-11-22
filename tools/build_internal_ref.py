"""Assemble docs/overview.html (Foresight Internal Reference) from hand-written fragments plus tables
generated from the repository's result files. Read-only on the repo except the one output file.

    python tools/build_internal_ref.py              # writes docs/overview.html
    python tools/build_internal_ref.py --out X.html # writes X.html instead (to diff before overwriting)
"""
import glob
import html
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Dk\Projects\1_Physical_AI\Foresight")
HERE = Path(__file__).resolve().parent
FRAG = HERE / "frag"
OUT = REPO / "docs" / "overview.html"


def esc(x):
    return html.escape(str(x), quote=True)


def fnum(v, nd=3, pct=False, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return esc(v)
    if pct:
        return f"{100 * f:+.1f}%" if sign else f"{100 * f:.1f}%"
    return f"{f:+.{nd}f}" if sign else f"{f:.{nd}f}"


def table(rows, header, num=(), cls="tbl dense", caption=None, hl=None):
    """rows: list of lists of already-formatted strings (escaped by the caller where needed)."""
    h = "".join(f'<th class="num">{esc(c)}</th>' if i in num else f"<th>{esc(c)}</th>" for i, c in enumerate(header))
    body = []
    for r_i, r in enumerate(rows):
        cl = ' class="hl"' if hl and hl(r_i, r) else ""
        tds = "".join(f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>" for i, c in enumerate(r))
        body.append(f"<tr{cl}>{tds}</tr>")
    cap = f'<p class="small tsrc">{caption}</p>' if caption else ""
    return f'<div class="{cls}"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{cap}'


# ---------------------------------------------------------------- result tables
MODEL_NAMES = {"cv": "Constant velocity", "cv_band": "CV + fixed band", "lgbm": "LightGBM, same 1,500 rows (untuned)",
               "lgbm_m50n100": "LightGBM, small-data setting m50 n100", "lgbm_all": "LightGBM, all rows (untuned)",
               "tabpfn": "TabPFN-3.5 default, 1,500 rows"}


def gen_claim_a():
    d = pd.read_csv(REPO / "results/claim_a_f4.csv")
    if "group" not in d.columns:
        return "<p>results/claim_a_f4.csv has an unexpected layout.</p>"
    out = []
    order = ["cv", "cv_band", "lgbm", "lgbm_m50n100", "lgbm_all", "tabpfn"]
    for grp, title in (("all", "All test rows"), ("moving", "Moving people only (speed > 0.2 m/s)"), ("standing", "Standing people only")):
        rows = []
        for direction in ("eth->hotel", "hotel->eth"):
            for m in order:
                s = d[(d.direction == direction) & (d.group == grp) & (d.model == m)]
                if not len(s):
                    continue
                n_seeds = s.seed.nunique() if "seed" in s else 1
                g = s.mean(numeric_only=True)
                sd = s.std(numeric_only=True) if n_seeds > 1 else None
                jc = fnum(g.joint90, pct=True) + (f" ± {100 * sd.joint90:.1f}" if sd is not None else "")
                er = f"{g.err:.3f}" + (f" ± {sd.err:.3f}" if sd is not None else "")
                rows.append([esc(direction.replace("->", " → ").replace("eth", "ETH").replace("hotel", "Hotel")),
                             esc(MODEL_NAMES.get(m, m)), fnum(int(g.n)) if "n" in g else "—", str(n_seeds), jc,
                             fnum(g.cov80, pct=True), f"{g.width80:.2f}", f"{g.is90:.2f}", f"{g.is80:.2f}",
                             fnum(g.point_miss05, pct=True), f"{g.pinball:.4f}", er])
        out.append(f"<h4>{esc(title)}</h4>" + table(
            rows, ["Direction", "Model", "n rows", "ctx seeds", "Joint cov. 5–95% box", "Per-axis cov. 10–90%",
                   "10–90% width (m)", "IS 90% (m)", "IS 80% (m)", "Point miss > 0.5 m", "Pinball", "Error (m)"],
            num=set(range(2, 12)), hl=lambda i, r: r[1].startswith("TabPFN")))
    return "".join(out) + ('<p class="small tsrc">Generated from <code>results/claim_a_f4.csv</code> (tools/claim_a.py, '
                           'written Sun 00:18, features f4, local TabPFN-3.5 default). "±" = spread over 5 context samples, '
                           'not a confidence interval. Nominal joint coverage of the 5–95% box: 80–90%.</p>')


def gen_claim_a_boot():
    p = REPO / "results/claim_a_boot_f4.csv"
    if not p.exists():
        return '<p class="small"><span class="st pend">pending</span> results/claim_a_boot_f4.csv not written yet.</p>'
    d = pd.read_csv(p)
    names = {"is80": "Interval score, 10–90% box (m)", "is90": "Interval score, 5–95% box (m)", "box_miss": "Joint 5–95% box miss"}
    rows = []
    for r in d.itertuples():
        pct = r.metric == "box_miss"
        f = (lambda v: f"{100 * v:+.1f} pts") if pct else (lambda v: f"{v:+.3f}")
        rows.append([esc(r.direction.replace("->", " → ").replace("eth", "ETH").replace("hotel", "Hotel")), esc(r.comparison),
                     esc(names.get(r.metric, r.metric)), f(r.diff), f"[{f(r.lo)}, {f(r.hi)}]", f"{int(r.persons)} / {int(r.rows)}",
                     f"<strong>{esc(r.verdict)}</strong>" if "better" in r.verdict else esc(r.verdict)])
    return table(rows, ["Direction", "Difference", "Metric", "Estimate", "95% person-bootstrap interval", "People / rows", "Verdict"],
                 num={3, 4, 5},
                 caption="Generated from <code>results/claim_a_boot_f4.csv</code> (tools/claim_a_boot.py): per test row, scores averaged "
                         "over the 5 context seeds; rows resampled by person (episode, agent_id), 5,000 times. Negative = TabPFN better.")


def gen_context_size():
    d = pd.read_csv(REPO / "results/context_size.csv")
    g = d.groupby(["model", "config", "n"]).agg(err=("err", "mean"), pin=("pinball", "mean"), cov=("joint90", "mean"),
                                                 wa=("width_along", "mean"), wc=("width_across", "mean"),
                                                 seeds=("seed", "nunique"), sec=("seconds", "mean")).reset_index()
    names = {"cv": "Constant velocity", "lgbm_all": "LightGBM all rows (untuned)", "lgbm_default": "LightGBM untuned",
             "lgbm_m50n100": "LightGBM m50 n100", "tabpfn_fast1_local": "TabPFN-3.5-Fast, 1 est. (local CPU)",
             "tabpfn_v35_api": "TabPFN-3.5 default, 8 est. (hosted API)"}
    rows = [[esc(names.get(r.model, r.model)), esc(r.config), fnum(int(r.n)), str(int(r.seeds)), f"{r.err:.3f}",
             "—" if np.isnan(r.pin) else f"{r.pin:.4f}", fnum(r.cov, pct=True), f"{r.wa:.2f} / {r.wc:.2f}", f"{r.sec:.1f}"]
            for r in g.itertuples()]
    return table(rows, ["Model", "Context", "Rows read", "Seeds", "Error +2 s (m)", "Pinball", "Joint cov. 5–95%",
                        "5–95% width along / across (m)", "Seconds per fit+predict"], num={2, 3, 4, 5, 6, 7, 8},
                 hl=lambda i, r: "TabPFN" in r[0],
                 caption="Generated from <code>results/context_size.csv</code> (tools/context_size.py; crowd_cross, f4, +2 s, "
                         "2,000 fixed test rows, split by episode 75/25 seed 0). \"retrieved\" = union of each test row's "
                         "nearest training rows (context.Bank), cut to the target size. Seconds are emulated-CPU wall time "
                         "(API rows: round trip).")


def gen_live_variants():
    d = pd.read_csv(REPO / "results/live_variants.csv")
    d["phase"] = np.where(d.setting == "A", "all", np.where(d.since >= 0, "after switch", "before switch"))
    g = d.groupby(["setting", "variant", "phase"]).agg(n=("err", "size"), err=("err", "mean"), cv=("cv_err", "mean"),
                                                         cov=("joint90", "mean"), w=("width", "mean"),
                                                         pin=("pinball", "mean"), t=("t_total", "mean"),
                                                         nctx=("n_ctx", "mean")).reset_index()
    g = g.sort_values(["setting", "phase", "variant"], ascending=[True, False, True])
    rows = [[esc(r.setting), esc(r.variant), esc(r.phase), fnum(int(r.n)), f"{r.err:.3f}", f"{r.cv:.3f}",
             fnum(r.cov, pct=True), f"{r.w:.2f}", f"{r.pin:.4f}", f"{r.nctx:.0f}", f"{r.t:.2f}"] for r in g.itertuples()]
    return table(rows, ["Setting", "Variant", "Phase", "Query rows", "Mean error (m)", "CV error (m)", "Joint cov. 5–95%",
                        "Width (m)", "Pinball", "Mean context rows", "s per moment"], num=set(range(3, 11)),
                 hl=lambda i, r: r[1] == "split",
                 caption="Generated from <code>results/live_variants.csv</code> (tools/live_variants.py, Sun 00:01). "
                         "Setting A = Hotel bank → first 6 ETH episodes (137 query rows over 44 moments, 83 people; real, noise 0). "
                         "Setting B = crowd_cross bank without erratic episodes → 6 new episodes (seeds 7000+) switching "
                         "calm → erratic at 20 s (636 query rows over 48 moments, 218 people). TabPFN = 3.5-Fast, 1 estimator, local.")


def gen_t3cp():
    d = pd.read_csv(REPO / "results/live_variants_cp.csv")
    d["phase"] = np.where(d.since >= 0, "after switch", "before switch")
    g = d.groupby(["variant", "phase"]).agg(n=("err", "size"), med=("err", "median"), mean=("err", "mean"),
                                             cov=("joint90", "mean"), w=("width", "mean"), isc=("is90", "mean"),
                                             pin=("pinball", "mean"), t=("t_moment_s", "mean"), dd=("d_used", "mean")).reset_index()
    g = g.sort_values(["phase", "variant"], ascending=[False, True])
    rows = [[esc(r.variant), esc(r.phase), fnum(int(r.n)), f"{r.med:.3f}", f"{r.mean:.3f}", fnum(r.cov, pct=True),
             f"{r.w:.2f}", f"{r.isc:.2f}", f"{r.pin:.4f}", f"{r.dd:+.3f}", f"{r.t:.2f}"] for r in g.itertuples()]
    return table(rows, ["Variant", "Phase", "Rows", "Median error (m)", "Mean error (m)", "Joint cov. 5–95%", "Width (m)",
                        "Interval score 90%", "Pinball", "Mean offset d (m)", "s per moment"], num=set(range(2, 11)),
                 hl=lambda i, r: r[0] == "split2k+offset",
                 caption="Generated from <code>results/live_variants_cp.csv</code> (tools/live_variants_cp.py, Sun 00:15; "
                         "Setting B exactly as live_variants). Paired person-bootstrap differences (split2k+offset minus "
                         "frozen2k+offset, after the switch) are quoted from RESULTS.md amendment 2, not recomputed here.")


def gen_t1():
    d = pd.read_csv(REPO / "results/t1_gate.csv")
    rows = [[("<strong>" + esc(r.cfg) + "</strong>") if r.learned else esc(r.cfg), "learned" if r.learned else "baseline",
             str(int(r.n)), fnum(r.success, pct=True), f"{r.time:.2f} [{r.time_lo:.2f}, {r.time_hi:.2f}]",
             f"{r.intrusion:.2f} [{r.intr_lo:.2f}, {r.intr_hi:.2f}]", fnum(r.collided, pct=True), f"{r.min_gap:.2f}",
             f"{r.stopped:.2f}"] for r in d.itertuples()]
    return table(rows, ["Config", "Kind", "Seeds", "Success", "Time to goal, s [95% CI]", "Intrusion s [95% CI]",
                        "Collision rate", "Min gap (m)", "Seconds stopped"], num=set(range(2, 9)),
                 hl=lambda i, r: r[1] == "learned",
                 caption="Generated from <code>results/t1_gate.csv</code> (tools/t1_gate.py, Sat 22:40). Default plaza "
                         "(data/crowd bank, untuned f3 filter: data/crowd has no kalman.json), seeds 1000–1099, switch on, "
                         "top speed 1.0 m/s. Intervals: marginal bootstrap per configuration.")


def gen_t2():
    d = pd.read_csv(REPO / "results/t2_cross.csv")
    rows = []
    for r in d.itertuples():
        rows.append([f"r{r.robot_repulsion:g} · i{r.ignore_robot:g} · p{int(r.people)}", str(int(r.n_seeds)),
                     "yes" if r.complete else "no", esc(r.best), fnum(r.best_mc_rate, pct=True), f"{r.best_intrusion_s:.2f}",
                     "<strong>yes</strong>" if r.qualifies else "no",
                     " / ".join("—" if pd.isna(x) else f"{100 * x:.0f}" for x in (r.stop_go_mc, r.cv_mc, r.reactive_mc, r.vo_mc)),
                     " / ".join("—" if pd.isna(x) else f"{x:.1f}" for x in (r.stop_go_time, r.cv_time, r.reactive_time, r.vo_time))])
    return table(rows, ["Cell (repulsion · ignore · people)", "Seeds", "Complete", "Best baseline", "Its moving-coll. rate",
                        "Its intrusion (s)", "Qualifies", "Moving-coll. % stop_go / cv / reactive / vo",
                        "Time s stop_go / cv / reactive / vo"], num={1, 4, 5},
                 hl=lambda i, r: r[0].startswith("r2 · i0.2 · p24"),
                 caption="Generated from <code>results/t2_cross.csv</code> (tools/t2_pick.py, Sun 00:10). Crossing crowd, f4, "
                         "seeds 4100–4119, baselines only (stop_go, cv m0.4, reactive m0.4, vo w1). Picker order: highest "
                         "repulsion, then lowest ignore share, then fewest people; first qualifying complete cell wins → "
                         "r2 · i0.2 · p24 (D-32).")


def gen_oracle():
    p = pd.read_csv(REPO / "results/oracle_gate_cross_points.csv")
    c = pd.read_csv(REPO / "results/oracle_gate_cross.csv")
    rows = [[esc(r.arm), f"{r.time_capped:.2f}", f"{r.min_gap:.3f}", fnum(r.mc_rate, pct=True)] for r in p.itertuples()]
    t1 = table(rows, ["Arm (planner:dial)", "Mean time_capped (s)", "Mean min gap (m)", "Moving-collision rate"],
               num={1, 2, 3}, hl=lambda i, r: r[0].startswith("oracle"),
               caption="Generated from <code>results/oracle_gate_cross_points.csv</code>: 20 paired seeds (4100–4119), cell "
                       "r2 · i0.2 · p24, crossing crowd, f4.")
    rows = [[esc(r.b), "min gap" if r.axis == "gap" else "moving coll.", esc(r.how), f"{r.diff:+.2f}",
             "—" if pd.isna(r.lo) else f"{r.lo:+.2f}", "—" if pd.isna(r.hi) else f"{r.hi:+.2f}", f"{r.p:.4f}",
             "<strong>beats</strong>" if r.beats else "does not beat"] for r in c.itertuples()]
    t2 = table(rows, ["Baseline point", "Safety axis", "How compared", "Time saved by oracle (s)", "95% lo", "95% hi",
                      "Share of resamples ≤ 0", "Verdict (> 2 s, CI excl. 0)"], num={3, 4, 5, 6},
               hl=lambda i, r: "does not" in r[7],
               caption="Generated from <code>results/oracle_gate_cross.csv</code> (tools/curve_compare.py --a oracle). Gate = "
                       "the oracle must beat every baseline point on both safety axes; highlighted rows are the failures.")
    return t1 + t2


def _closed(pattern):
    fs = glob.glob(str(REPO / "results/closed" / f"*{pattern}.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def gen_t4off():
    rows = []
    for tag, label in (("t4off-off", "OFF (scored only, eta 0)"), ("t4off-row", "(i) per row, eta 0.02"),
                       ("t4off-batch", "(ii) per batch, eta_b 0.05 / 2 s")):
        d = _closed(tag)
        if not len(d):
            continue
        for p, s in d.groupby("planner"):
            rows.append([esc(label), esc(p), str(s.seed.nunique()), fnum(s.inloop_cov90d_post.mean(), pct=True),
                         fnum(s.inloop_cov90_post.mean(), pct=True), f"{s.seconds_stopped.mean():.2f}",
                         f"{s.freeze_rate.mean():.3f}", f"{s.offset_final.mean():+.3f}", f"{s.time_capped.mean():.2f}",
                         f"{s.pred_ms_per_plan.mean():.0f}"])
    return table(rows, ["Offset rule", "Planner", "Seeds", "Cov. after switch, 5–95% + offset (cov90d)", "Raw 5–95% cov. after",
                        "Seconds stopped", "Freeze rate", "Final offset d (m)", "Time capped (s)", "ms per plan"],
                 num=set(range(2, 10)), hl=lambda i, r: r[1] == "tabpfn_split",
                 caption="Generated from <code>results/closed/*t4off-{off,row,batch}.csv</code> (tools/t4_offset.ps1; seeds "
                         "5100–5105; T5 config, box_q 0.15, margin 0.2). The gate is judged on tabpfn_split only.")


def gen_t4sm():
    a, b = _closed("t4sm-off"), _closed("t4sm-on")
    if not len(a) or not len(b):
        return "<p>No T4 smoothing files found.</p>"
    rows = []
    for p in ("tabpfn_split", "tabpfn", "lgbm_all", "cv"):
        x, y = a[a.planner == p].set_index("seed"), b[b.planner == p].set_index("seed")
        s = x.index.intersection(y.index)
        x, y = x.loc[s], y.loc[s]
        hc0, hc1 = x.heading_change_per_m.mean(), y.heading_change_per_m.mean()
        dt_ = y.time_capped.mean() - x.time_capped.mean()
        di = y.intrusion_s.mean() - x.intrusion_s.mean()
        dm = (y.moving_collision_steps > 0).mean() - (x.moving_collision_steps > 0).mean()
        ok = p != "cv" and (hc1 / hc0 - 1) <= -0.30 and dt_ <= 2 and di <= 1 and dm <= 0.1
        rows.append([esc(p), str(len(s)), f"{hc0:.3f} → {hc1:.3f}", f"{100 * (hc1 / hc0 - 1):+.0f}%", f"{dt_:+.2f}",
                     f"{di:+.2f}", f"{dm:+.2f}", "—" if p == "cv" else ("pass" if ok else "<strong>fail</strong>")])
    return table(rows, ["Planner", "Paired seeds", "Heading change per m (rad/m)", "Change", "Δ time (s)", "Δ intrusion (s)",
                        "Δ moving-coll. rate", "D-28 gate"], num={1, 3, 4, 5, 6},
                 caption="Generated now from <code>results/closed/*t4sm-{off,on}.csv</code> (seeds 5110–5119), paired on "
                         "the seeds present in both files. Note: tools/t4_smooth.ps1 ran every planner, the learned ones "
                         "included, at margin 0.4, while T5's learned arms run at the D-7 margin of 0.2.")


def gen_tier():
    d = pd.read_csv(REPO / "results/tier_seeds.csv")
    g = d.groupby(["setup", "model", "est", "context"]).agg(err=("err", "mean"), err_sd=("err", "std"), pin=("pinball", "mean"),
                                                            cov=("joint90", "mean"), w=("width80", "mean"),
                                                            n=("seed", "nunique")).reset_index()
    rows = [[esc(r.setup.replace("->", " → ")), esc(r.model), esc(r.est), fnum(int(r.context)), str(int(r.n)),
             f"{r.err:.3f} ± {r.err_sd:.3f}" if not np.isnan(r.err_sd) else f"{r.err:.3f}", f"{r.pin:.3f}",
             fnum(r.cov, pct=True), f"{r.w:.2f}"] for r in g.itertuples()]
    return table(rows, ["Setup", "Model", "Estimators", "Context rows", "Seeds", "Error +2 s (m)", "Pinball",
                        "Joint cov. 5–95%", "10–90% width (m)"], num={3, 4, 5, 6, 7, 8},
                 caption="Generated from <code>results/tier_seeds.csv</code> (tools/tier_seeds.py, hosted API, Sat 22:02). "
                         "<span class=\"st sup\">superseded</span> features f1 (heading leak, untuned filter). Kept because "
                         "D-5/D-18 were decided on it; tier_seeds was then dropped (D-29).")


def gen_api():
    d = pd.read_csv(REPO / "results/api_usage.csv")
    rows = [[esc(r.date_utc), esc(r.run), "—" if pd.isna(r.calls) else str(int(r.calls)), fnum(int(r.credits)), esc(r.note)]
            for r in d.itertuples()]
    return table(rows, ["Date (UTC)", "Run", "Calls", "Credits", "Note"], num={2, 3},
                 caption="Copied from <code>results/api_usage.csv</code> at build time.")


def gen_t5_status():
    fs = sorted(glob.glob(str(REPO / "results/closed" / "*t5-*.csv")))
    tags = sorted({re.sub(r"^\d{8}-\d{6}-", "", Path(f).stem) for f in fs if not Path(f).name.startswith("robotfeat_")})
    if not tags:
        return "<p>No T5 result files yet.</p>"
    n_ext = sum(t.endswith("-ext") for t in tags)
    return ('<p class="small">T5 analysed on 20 paired seeds (3000–3019): <code>results/t5_primary.csv</code>, '
            '<code>results/inloop_claimb.csv</code>, <code>results/t5_oracle_matchedtime.csv</code>; RESULTS "Closed loop T5 '
            'results" (fce98a6). A 40-seed extension (same arms and rules) was declared at 02:20 (3226e4f); the 20-seed '
            'analysis stays the confirmatory result; the 40-seed tables follow as an update, computed by the same scripts and '
            'rules (RESULTS "Extension to 40 seeds", <code>tools/t5_extend.py</code>). At page build: '
            f'{len(tags)} T5 result tags in <code>results/closed/</code>, {n_ext} of them <code>-ext</code>.</p>')


# ---------------------------------------------------------------- walkthrough appendix
WT = REPO / "docs" / "walkthrough"


def _csv_table(name, caption, round_to=3):
    p = WT / name
    if not p.exists():
        return f"<p class=\"small\">{esc(name)}: not present.</p>"
    d = pd.read_csv(p)
    rows = []
    num = set()
    for j, c in enumerate(d.columns):
        if pd.api.types.is_numeric_dtype(d[c]):
            num.add(j)
    for r in d.itertuples(index=False):
        cells = []
        for j, v in enumerate(r):
            if j in num:
                cells.append("—" if pd.isna(v) else (f"{v:.{round_to}f}" if isinstance(v, float) else esc(v)))
            else:
                cells.append("" if pd.isna(v) else esc(v))
        rows.append(cells)
    return table(rows, list(d.columns), num=num, caption=f"<code>docs/walkthrough/{esc(name)}</code>. {caption}")


def _json(name):
    p = WT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _flat(dct, prefix=""):
    out = {}
    for k, v in dct.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _kv_table(dct, caption, keys=None):
    rows = []
    for k, v in _flat(dct).items():
        if keys and k not in keys:
            continue
        if isinstance(v, list):
            v = ", ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in v)[:200]
        elif isinstance(v, float):
            v = f"{v:.4f}"
        rows.append([esc(k), esc(v)])
    return table(rows, ["Key", "Value"], cls="tbl dense idfirst", caption=caption)


def gen_walk():
    meta = _json("meta.json")
    planner = meta.get("planner", "?")
    final = planner == "tabpfn_split"
    status = ('<span class="st fin">current</span> Regenerated on the final closed-loop configuration '
              '(<code>tabpfn_split</code>, T5 cell, switch into erratic at 5 s). This is a demonstration seed (6001), '
              'not a T5 seed; the moment is picked by the fixed rule in <code>tools/walkthrough.py: choose_moment</code> '
              '(earliest replan after the switch with the scene model active, ≥ 3 box vetoes, and a moving person within '
              '3 m whose own box vetoes a candidate). Timings were measured while T5 occupied the CPU.' if final else
              '<span class="st pend">pending regeneration</span> These files still come from the earlier '
              f'<code>{esc(planner)}</code> planner (mixed recent / kNN / fill context, 16 people, box_q '
              f'{esc(meta.get("box_q"))}); tools/walkthrough.py is being rerun on the final T5 configuration '
              '(tabpfn_split, 24 people, switch into erratic at 5 s, box_q 0.15). Treat the numbers below as '
              '<span class="st sup">superseded</span> until meta.json says <code>tabpfn_split</code>.')
    head = (f'<div class="verdict"><span class="lab">Source of this appendix</span><p>{status}</p>'
            f'<p class="small">meta.json: planner <code>{esc(planner)}</code>, seed {esc(meta.get("seed"))}, decision t = '
            f'{esc(meta.get("t"))} s, person P = id {esc(meta.get("P"))} ({fnum(meta.get("P_dist_m"), 2)} m from the robot, '
            f'{fnum(meta.get("P_speed"), 2)} m/s), commit {esc(meta.get("commit"))}, features {esc(meta.get("features_version"))}, '
            f'generated {esc(meta.get("generated"))}.</p></div>')

    def fig(name, alt, cap):
        if not (WT / name).exists():
            return ""
        b = (WT / name).read_bytes()[:24]
        w, h = int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")     # PNG IHDR
        return (f'<figure><div class="plate"><img src="walkthrough/{esc(name)}" alt="{esc(alt)}" loading="lazy" '
                f'width="{w}" height="{h}"></div>'
                f'<figcaption>{cap}</figcaption></figure>')

    parts = [head, "<h3>A.0 Run settings (meta.json)</h3>", _kv_table(meta, "<code>docs/walkthrough/meta.json</code>")]
    parts += ["<h3>A.1 Scene (S0)</h3>", fig("s1_scene.png", "Top-down scene at the decision moment",
                                              "The scene at the decision moment: people (blue), the robot (violet), its goal, pillars.")]
    parts += ["<h3>A.2 Sensing: true, noisy and filtered positions of P (S1)</h3>",
              _csv_table("s2_sensing.csv", "True positions (simulator), noisy observations (0.1 m), forward Kalman filter output. "
                                           "Only the filtered values enter the model."),
              fig("s2_sensing.png", "P's true, noisy and filtered track", "P's last seconds: truth, noisy observations, filtered track.")]
    parts += ["<h3>A.3 The feature row of P (S2): all 39 model columns</h3>",
              _csv_table("s3_features.csv", "Every model input column for P at both horizons, in P's own frame.")]
    cc = _json("s4_context_counts.json")
    parts += ["<h3>A.4 Context TabPFN read (S3)</h3>",
              _kv_table({k: v for k, v in cc.items() if k != "knn_feats"}, "<code>docs/walkthrough/s4_context_counts.json</code>: rows by source at the refit that served this decision."),
              _csv_table("s4_context_excerpt.csv", "A few context rows with their labels (ddx along, ddy across)."),
              fig("s4_context.png", "Context rows coloured by source", "The context rows, coloured by source.")]
    parts += ["<h3>A.5 TabPFN quantiles for P (S4)</h3>",
              _csv_table("s5_quantiles.csv", "Deviation from the constant-velocity point, metres, P's frame; \"actual\" = what happened."),
              fig("s5_person_box.png", "P's predicted boxes against the truth", "P's boxes at +1 s and +2 s against the true future position."),
              fig("s6_all_boxes.png", "Everyone's predicted boxes", "Everyone at the same moment: predicted boxes and true future positions.")]
    occ = _json("s7_occupancy_scores.json")
    if occ:
        rows = [[f"+{esc(h)} s", f"{v.get('brier', np.nan):.5f}", f"{v.get('brier_all_zero', np.nan):.5f}", f"{v.get('auc', np.nan):.3f}",
                 str(v.get("true_cells")), str(v.get("cells")), f"{v.get('expected_people', np.nan):.2f}",
                 f"{v.get('p_at_truth_mean', np.nan):.3f}"] for h, v in occ.items()]
        parts += ["<h3>A.6 Occupancy at this moment (S6)</h3>",
                  table(rows, ["Horizon", "Brier", "Brier of all-zero map", "ROC AUC", "True cells", "Cells", "Expected people",
                               "Mean p at truth"], num=set(range(1, 8)),
                        caption="<code>docs/walkthrough/s7_occupancy_scores.json</code>. One moment only: indicative, not the S6 metric."),
                  fig("s7_occupancy.png", "Occupancy heat map", "Occupancy grid from the same quantiles (0.25 m cells).")]
    parts += ["<h3>A.7 Candidates (S7)</h3>",
              _csv_table("s8_candidates.csv", "All candidate moves with worst gap, progress in 2 s, veto reason and which person's box blocked it."),
              fig("s8_planner.png", "Candidate moves, vetoed and allowed", "The planner at this moment: vetoed (red), allowed (green), chosen (thick).")]
    s9 = _json("s9_metrics.json")
    parts += ["<h3>A.8 Episode outcome (S8 and metrics)</h3>", _kv_table(s9, "<code>docs/walkthrough/s9_metrics.json</code>: the whole episode for this planner and seed."),
              fig("s9_ego_path.png", "Robot path over the episode", "The robot's path over the whole episode.")]
    ep = _json("episode_stats.json")
    if ep:
        flat = {k: v for k, v in ep.items() if not isinstance(v, (list, dict))}
        parts += [_kv_table(flat, "<code>docs/walkthrough/episode_stats.json</code> (scalar fields).")]
        if isinstance(ep.get("veto_reasons_all_plans"), dict):
            vr = ep["veto_reasons_all_plans"]
            parts += [table([[esc(k), fnum(v)] for k, v in vr.items()], ["Veto reason, all plans of the episode", "Candidates"],
                            num={1}, cls="tbl dense idfirst")]
        cbr = ep.get("context_by_refit")
        if isinstance(cbr, list) and cbr:
            keys = sorted({k for r in cbr for k in r if k != "t"})
            rows = [[f"{r.get('t', np.nan):.1f}"] + [fnum(r.get(k, 0)) for k in keys] for r in cbr]
            parts += [table(rows, ["Refit t (s)"] + keys, num=set(range(0, len(keys) + 1)), cls="tbl dense idfirst",
                            caption="Context rows by source at every refit of the episode.")]
    return "".join(parts)


# ---------------------------------------------------------------- data section (section 4)
DATASETS = ["eth", "hotel", "crowd", "crowd_c2e", "crowd_cross", "crowd_cross_t2", "toy"]


def _f4_cache(name):
    fs = sorted(glob.glob(str(REPO / "data" / name / "rows_f4_*.parquet")))
    return Path(fs[0]) if fs else None


def gen_data_tracks():
    """Per dataset, from tracks.csv + meta.json (+ obstacles.csv): what the raw/simulated tracks hold."""
    rows = []
    for name in DATASETS:
        d = REPO / "data" / name
        if not (d / "tracks.csv").exists():
            continue
        t = pd.read_csv(d / "tracks.csv")
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        eps = m.get("episodes", {})
        ppl = t[t.kind == "person"]
        n_sw = sum(1 for v in eps.values() if v.get("regime_switch_t") is not None)
        sw_t = [v["regime_switch_t"] for v in eps.values() if v.get("regime_switch_t") is not None]
        starts = pd.Series([v["regime_before"] for v in eps.values()]).value_counts()
        cfg = m.get("config", {})
        cfg_s = ", ".join(f"{k} {v}" for k, v in cfg.items() if k in ("n_people", "goal_mode", "robot_repulsion", "ignore_robot"))
        ob = pd.read_csv(d / "obstacles.csv") if (d / "obstacles.csv").exists() else pd.DataFrame()
        rows.append([f"<code>{esc(name)}</code>", esc(m.get("sim", "?")), f"{float(m['dt']):g} s", fnum(t.episode.nunique()),
                     fnum(len(ppl)), fnum(ppl.groupby(["episode", "agent_id"]).ngroups), fnum(ppl.agent_id.nunique()),
                     "yes" if (t.is_ego == 1).any() else "no", fnum(len(ob)),
                     (f"{n_sw} ({min(sw_t):g}–{max(sw_t):g} s)" if n_sw else "0"),
                     (f"{100 * ppl['collided'].mean():.2f}%" if "collided" in ppl else "—"),
                     esc(", ".join(f"{k} {v}" for k, v in starts.items())), esc(cfg_s) if cfg_s else "—"])
    return table(rows, ["Dataset", "meta.sim", "dt", "Episodes", "Person rows", "Person tracks (episode, id)", "Distinct ids",
                        "Robot", "Obstacles", "Episodes with a switch", "Collided (person rows)", "Start regime (episodes)", "meta.config knobs"],
                 num={3, 4, 5, 6, 8, 10},
                 caption="Generated at build time from <code>data/&lt;name&gt;/tracks.csv</code>, <code>meta.json</code> and "
                         "<code>obstacles.csv</code> (read-only). \"Person tracks\" counts (episode, agent_id) pairs: ETH/Hotel "
                         "people who cross a 60 s window edge count twice; in cross mode every entrant gets a new id "
                         "(<code>crowd.py: CrowdSim._respawn</code>). Distinct ids in the simulator repeat across episodes "
                         "(ids restart at 0 each episode). No <code>config</code> in a meta.json means the set predates "
                         "the T2 knobs (defaults reproduce the original crowd, <code>CrowdConfig</code> comment).")


def gen_data_rows():
    """Per f4 row cache: what the feature rows hold (counts per horizon, labels, padding)."""
    rows = []
    for name in DATASETS:
        p = _f4_cache(name)
        if p is None:
            rows.append([f"<code>{esc(name)}</code>", "no f4 cache", *["—"] * 9])
            continue
        d = pd.read_parquet(p)
        h = d.groupby(d.h.round(2)).size()
        h2 = d[np.isclose(d.h, 2.0)]
        ms = h2["mode"].value_counts(normalize=True)
        unk = (d["mode"] == "unknown").mean()
        lab = " / ".join(f"{100 * ms.get(k, 0):.1f}" for k in ("straight", "stop", "turn_left", "turn_right"))
        pad = " / ".join(f"{100 * (d[c] >= 99).mean():.1f}" for c in ("n0_dist", "n2_dist", "robot_dist", "obst_dist"))
        rows.append([f"<code>{esc(name)}</code>", f"<code>{esc(p.name)}</code>", fnum(len(d)),
                     esc(" · ".join(f"{k:g} s: {v:,}" for k, v in h.items())), fnum(d.episode.nunique()),
                     fnum(d.groupby(["episode", "agent_id"]).ngroups), lab, f"{100 * unk:.1f}%", pad,
                     f"{h2.ddx.std():.2f} / {h2.ddy.std():.2f}", esc(", ".join(d.features_version.unique()))])
    return table(rows, ["Dataset", "f4 cache file", "Rows", "Rows per horizon", "Episodes with rows", "People with rows",
                        "Labels at +2 s: straight / stop / left / right (%)", "unknown (all h)",
                        "Padded 99: n0 / n2 / robot / obstacle (%)", "Target sd at +2 s, ddx / ddy (m)", "Version"],
                 num={2, 4, 5, 7, 9},
                 caption="Generated at build time from <code>data/&lt;name&gt;/rows_f4_*.parquet</code>, the caches written by "
                         "<code>run_offline.rows_for()</code> (file name = features version, sensor noise, horizons, Kalman q and r). "
                         "Real data: noise 0, horizons 1.2 / 2 / 4.8 s; simulation: noise 0.1 m, horizons 1 / 2 s. "
                         "<code>crowd</code>'s cache uses the untuned q 2, r 0.1 (no kalman.json); <code>crowd_c2e</code> and "
                         "<code>toy</code> have no f4 cache (crowd_c2e only unversioned f1-era caches). Labels at +2 s never "
                         "include <code>unknown</code>, because a +2 s row needs the person 2 s later.")


def gen_t5_bank():
    """The T5 / walkthrough bank: crowd_cross_t2 without any episode that is erratic before or after a switch."""
    p = _f4_cache("crowd_cross_t2")
    mp = REPO / "data/crowd_cross_t2/meta.json"
    if p is None or not mp.exists():
        return "<p class=\"small\">crowd_cross_t2 cache not present.</p>"
    d = pd.read_parquet(p, columns=["episode", "h", "mode"])
    em = json.loads(mp.read_text(encoding="utf-8"))["episodes"]
    bad = {int(e) for e, v in em.items() if "erratic" in (v.get("regime_before"), v.get("regime_after"))}
    keep = d[~d.episode.isin(bad)]
    ms = keep["mode"].value_counts(normalize=True)
    return (f'<p class="small tsrc">Recomputed at build time from <code>{esc(p.name)}</code> and <code>meta.json</code>: '
            f'{len(em) - len(bad)} of {len(em)} episodes have no erratic regime before or after a switch; they hold '
            f'{len(keep):,} of {len(d):,} rows ({(np.isclose(keep.h, 2.0)).sum():,} at +2 s). Move shares in that bank: '
            + ", ".join(f"{esc(k)} {100 * v:.1f}%" for k, v in ms.items())
            + '. Same filter as <code>run_closed.py --bank-exclude-regime erratic</code>. '
            + (lambda s4: f'The walkthrough <code>meta.json</code> (s4) records {s4.get("frozen_episodes", "?")} episodes / '
                          f'{s4.get("bank_rows", "?"):,} rows for the bank it used.' if s4 else '')(_json("meta.json").get("s4", {}))
            + '</p>')


FEAT_DOC = {   # name -> (meaning, unit, frame)
    "h_f": ("horizon the row asks about", "s", "—"),
    "speed": ("filtered speed |v|", "m/s", "scalar"),
    "a_lon": ("acceleration along the heading, (v(t) − v(t − k·dt)) / (k·dt)", "m/s²", "agent"),
    "a_lat": ("acceleration across the heading (+ = left)", "m/s²", "agent"),
    "turn_rate": ("heading change over one history step / (k·dt); 0 when speed ≤ 0.2", "rad/s", "scalar (+ = left)"),
    "radius": ("body radius from tracks.csv", "m", "scalar"),
}
for _j in range(1, 6):
    FEAT_DOC[f"hist_x{_j}"] = (f"filtered position {_j} history step(s) ago minus now, along", "m", "agent")
    FEAT_DOC[f"hist_y{_j}"] = (f"same, across (+ = left)", "m", "agent")
for _n, _w in enumerate(("nearest", "2nd nearest", "3rd nearest")):
    FEAT_DOC[f"n{_n}_x"] = (f"{_w} other person (robot excluded): offset along", "m", "agent")
    FEAT_DOC[f"n{_n}_y"] = (f"{_w}: offset across", "m", "agent")
    FEAT_DOC[f"n{_n}_vx"] = (f"{_w}: velocity relative to the person, along", "m/s", "agent")
    FEAT_DOC[f"n{_n}_vy"] = (f"{_w}: relative velocity, across", "m/s", "agent")
    FEAT_DOC[f"n{_n}_dist"] = (f"{_w}: centre distance", "m", "scalar")
FEAT_DOC.update({
    "robot_x": ("robot centre offset, along", "m", "agent"), "robot_y": ("robot centre offset, across", "m", "agent"),
    "robot_dist": ("distance to the robot centre", "m", "scalar"),
    "robot_closing": ("−(d · (v_robot − v)) / |d|: approach speed, + = closing", "m/s", "scalar"),
    "obst_x": ("vector to the nearest pillar surface or wall, along", "m", "agent (padding unrotated, f2)"),
    "obst_y": ("same, across", "m", "agent (padding unrotated, f2)"),
    "obst_dist": ("distance to that surface", "m", "scalar"),
    "density_3m": ("other people (not the robot) within 3 m", "count", "scalar"),
})


def _pad_rule(name):
    if name.startswith("n") and name[1].isdigit():
        return {"x": "99", "y": "0", "vx": "0", "vy": "0", "dist": "99"}[name.split("_", 1)[1]], f"n{name[1]}_dist"
    if name.startswith("robot_"):
        return {"x": "99", "y": "0", "dist": "99", "closing": "0"}[name.split("_", 1)[1]], "robot_dist"
    if name.startswith("obst_"):
        return {"x": "99", "y": "0", "dist": "99"}[name.split("_", 1)[1]], "obst_dist"
    if name.startswith("hist_"):
        return "row dropped", None
    if name == "density_3m":
        return "0 (a count)", None
    return "never missing", None


def gen_feature_table():
    import sys
    sys.path.insert(0, str(REPO))
    from foresight.core.features import feature_names
    names = ["h_f"] + feature_names()[1:]
    assert len(names) == 39 and set(names) == set(FEAT_DOC), "FEAT_DOC out of sync with features.feature_names()"
    caches = {n: pd.read_parquet(p) for n in ("eth", "hotel", "crowd_cross_t2") if (p := _f4_cache(n)) is not None}
    rows = []
    for i, n in enumerate(names, 1):
        meaning, unit, frame = FEAT_DOC[n]
        code, key = _pad_rule(n)
        shares = " / ".join(f"{100 * (c[key] >= 99).mean():.1f}" if key else "—" for c in caches.values())
        rows.append([str(i), f"<code>{esc(n)}</code>", esc(meaning), esc(unit), esc(frame), esc(code), shares, "no"])
    return table(rows, ["#", "Column", "Meaning", "Unit", "Frame", "When absent", "Padded share % (" + " / ".join(caches) + ")",
                        "Categorical"], num={0}, cls="tbl dense idfirst",
                 caption="Generated at build time: names and order from <code>foresight/core/features.py: feature_names()</code> "
                         "(the builder asserts 39 and that every name is documented); padding codes from <code>build_rows()</code> "
                         "(<code>FAR = 99.0</code>); padded shares measured on the f4 caches. History step k = max(1, round(0.2/dt)) "
                         "steps: 0.2 s in simulation, 0.4 s on ETH/Hotel (Python rounds 0.5 to 0), so the five history samples cover "
                         "1.0 s in simulation and 2.0 s on real data. No column is declared categorical (no "
                         "<code>categorical_features_indices</code> anywhere in the repo); <code>h_f</code> takes 2–3 values "
                         "and <code>radius</code> one value per dataset, both passed as numbers.")


def gen_worked_row():
    p = WT / "s3_features.csv"
    if not p.exists():
        return "<p class=\"small\">docs/walkthrough/s3_features.csv not present.</p>"
    d = pd.read_csv(p)
    meta = _json("meta.json")
    rows = []
    for i, r in enumerate(d.itertuples(index=False), 1):
        v1, v2 = getattr(r, "_2"), getattr(r, "_3")
        cell = f"{v1:.3f}" if abs(v1 - v2) < 1e-12 else f"{v1:.3f} / {v2:.3f}"
        unit = FEAT_DOC.get(r.feature, ("", "", ""))[1]
        rows.append([str(i), f"<code>{esc(r.feature)}</code>", esc(r.group), cell, esc(unit), esc(r.meaning)])
    t = table(rows, ["#", "Column", "Group", "Value (h 1 s / 2 s if different)", "Unit", "Meaning (walkthrough's own label)"],
              num={0, 3}, cls="tbl dense idfirst",
              caption=f"Generated at build time from <code>docs/walkthrough/s3_features.csv</code>: person P = id "
                      f"{esc(meta.get('P'))}, seed {esc(meta.get('seed'))}, t = {esc(meta.get('t'))} s, features "
                      f"{esc(meta.get('features_version'))}, filter q {esc(meta.get('kalman_q'))}, r {esc(meta.get('kalman_r'))}, "
                      f"commit {esc(meta.get('commit'))}. One live query row per horizon; only <code>h_f</code> differs "
                      "between the two.")
    q = WT / "s5_quantiles.csv"
    if q.exists():
        s = pd.read_csv(q)
        a = s.groupby(["horizon", "axis"]).actual.first()
        t += ('<p class="small tsrc">The label this row later received (<code>s5_quantiles.csv</code>, column "actual" = where P really went in the simulator, in P&#39;s frame, relative to the walk-straight point): '
              + "; ".join(f"{esc(h)} {esc(ax.split(' ')[0])} {v:+.3f} m" for (h, ax), v in a.items())
              + ". Along: " + ", ".join(f"{esc(h)} {abs(v):.2f} m {'short of' if v < 0 else 'beyond'}"
                                       for (h, ax), v in a.items() if ax.startswith("along"))
              + " the walk-straight point (row: <code>a_lon</code> "
              + f"{float(d.loc[d.feature == 'a_lon'].iloc[0, 2]):+.2f} m/s²). Across: "
              + ", ".join(f"{esc(h)} {v:+.2f} m ({'left' if v > 0 else 'right'})" for (h, ax), v in a.items() if ax.startswith("across"))
              + ".</p>")
    return t


GEN = {"claim_a": gen_claim_a, "claim_a_boot": gen_claim_a_boot, "context_size": gen_context_size, "live_variants": gen_live_variants, "t3cp": gen_t3cp,
       "t1": gen_t1, "t2": gen_t2, "oracle": gen_oracle, "t4off": gen_t4off, "t4sm": gen_t4sm, "tier": gen_tier,
       "api": gen_api, "t5_status": gen_t5_status, "walk": gen_walk,
       "data_tracks": gen_data_tracks, "data_rows": gen_data_rows, "t5_bank": gen_t5_bank,
       "feature_table": gen_feature_table, "worked_row": gen_worked_row}


def main(out=OUT):
    # Section fragments in file-name order. Optional fragments (e.g. p05b_validation.html, written by another session)
    # are picked up when present and skipped otherwise: "p05_exp" < "p05b_validation" < "p06_decisions", because "_" < "b".
    parts = sorted(FRAG.glob("p*.html"))
    names = [p.name for p in parts]
    if "p05b_validation.html" in names:
        assert names.index("p05_exp.html") + 1 == names.index("p05b_validation.html"), names
    text = "".join(p.read_text(encoding="utf-8") for p in parts)

    def nav_opt(m):
        """{{NAVOPT:fragment.html}} in the contents rail: one <li> linking the fragment's first <h2 id=...>, or nothing
        when the fragment does not exist. Label = section number (leading zeros dropped) + heading text."""
        p = FRAG / m.group(1)
        if not p.exists():
            return ""
        h = re.search(r'<h2 id="([^"]+)"[^>]*>(.*?)</h2>', p.read_text(encoding="utf-8"), re.S)
        if not h:
            return ""
        sec = re.search(r'<span class="sec">(.*?)</span>', h.group(2))
        title = re.sub(r"<[^>]+>", "", re.sub(r'<span class="sec">.*?</span>', "", h.group(2))).strip()
        num = (sec.group(1).lstrip("0") + " ") if sec else ""
        return f'<li><a href="#{esc(h.group(1))}">{esc(num)}{esc(title)}</a></li>'

    def file_sub(m):
        return (FRAG / m.group(1)).read_text(encoding="utf-8")

    def gen_sub(m):
        try:
            return GEN[m.group(1)]()
        except Exception as e:  # keep the page buildable; show the problem inline
            return f'<p class="small"><strong>Table {esc(m.group(1))} could not be generated:</strong> {esc(e)}</p>'

    text = re.sub(r"[ \t]*\{\{NAVOPT:([\w.\-]+)\}\}\n?", lambda m: (lambda s: f"    {s}\n" if s else "")(nav_opt(m)), text)
    text = re.sub(r"\{\{FILE:([\w.\-]+)\}\}", file_sub, text)
    text = re.sub(r"\{\{GEN:(\w+)\}\}", gen_sub, text)
    css = (FRAG / "style.css").read_text(encoding="utf-8")
    text = text.replace("{{CSS}}", css)
    out = Path(out)
    out.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", out, len(text.encode("utf-8")), "bytes")


if __name__ == "__main__":
    import sys
    # optional: --out PATH writes elsewhere (e.g. a temp file to diff against docs/overview.html)
    main(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT)
