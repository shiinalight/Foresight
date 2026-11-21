"""3D replay in the browser, styled like a miniature model: the street scene or the doorway scene.

    python tools/street3d.py                                   # episode 0 of data/street
    python tools/street3d.py --data data/street --episode 2 --open
    python tools/street3d.py --data data/doorway_test --episode 1001 --open   # doorway: two rooms, one door

    python tools/street3d.py --data data/street_test --episode 0 --predict --open
        # + TabPFN's 80% range at +2 s under every agent (held-out episode; context from data/street)

The scene is read from meta.json ("sim"); one template draws both.

Writes results/street3d_ep<N>.html (doorway: results/doorway3d_ep<N>.html): one self-contained page (three.js from a CDN, the episode's
data inlined). Drag to orbit, scroll to zoom; camera buttons for overview, follow-the-robot and
street level. Scripted events show as labels above the agents involved and as marks on the timeline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.domains import street as S  # noqa: E402

ACTS = ["walk", "wait", "cross", "jaywalk", "phone_slow", "phone", "stop", "drive", "queue", "door", "robot", "order", "wait_door"]
TYPE_OF_KIND = {"person": "pedestrian", "robot": "robot", "car": "car", "cyclist": "cyclist"}
TEMPLATE = Path(__file__).with_name("street3d_template.html")


def scene_of(meta: dict) -> str:
    return "doorway" if str(meta.get("sim", "")).startswith("doorway") else "street"


def export(tracks: pd.DataFrame, meta: dict, episode: int) -> dict:
    tr = tracks[tracks.episode == episode].sort_values(["agent_id", "t"]).copy()
    dt = float(meta["dt"])
    em = meta["episodes"][str(episode)]
    scene = scene_of(meta)
    if "agent_type" not in tr:                                   # doorway: kind only, no box size or heading
        tr["agent_type"] = tr["kind"].map(TYPE_OF_KIND).fillna("pedestrian")
    for c in ("length", "width"):
        if c not in tr:
            tr[c] = 0.0
    if "heading" not in tr:                                      # from velocity, held while standing still
        h = pd.Series(np.arctan2(tr.vy, tr.vx), index=tr.index).where(np.hypot(tr.vx, tr.vy) > 0.15)
        tr["heading"] = h.groupby(tr.agent_id).transform(lambda s: s.ffill().bfill()).fillna(0.0)
    agents = []
    for aid, g in tr.groupby("agent_id"):
        k = np.round(g.t.to_numpy() / dt).astype(int)
        k0, n = int(k.min()), int(k.max() - k.min() + 1)

        def col(values, nd=2):
            out = [None] * n
            for i, v in zip(k - k0, values):
                out[i] = round(float(v), nd)
            return out
        act = g["activity"].fillna("walk") if "activity" in g else pd.Series(["walk"] * len(g))
        first = g.iloc[0]
        agents.append(dict(
            id=int(aid), type=first.agent_type, len=round(float(first.length), 2), wid=round(float(first.width), 2),
            r=round(float(first.radius), 2), k0=k0,
            x=col(g.x), y=col(g.y), h=col(g.heading, 3), v=col(np.hypot(g.vx, g.vy)),
            a=[ACTS.index(s) if s in ACTS else 0 for s in act]))
    n_frames = int(round(tr.t.max() / dt)) + 1
    rush_t = em.get("regime_switch_t")
    if scene == "doorway":                                       # the wall is drawn from the map, not the circle chain
        obstacles = []
        events = [dict(t=rush_t, name="rush", agent_id=-1)] if rush_t is not None else []
    else:
        obstacles, events = [[x, y, r] for x, y, r in S.OBSTACLES], em["events"]
    return dict(scene=scene, dt=dt, n=n_frames, episode=episode, seed=em["seed"], bounds=meta["bounds"], map=meta["map"],
                obstacles=obstacles, events=events, rush_t=rush_t, acts=ACTS, agents=agents)


def predictions(scene, episode, h=2.0):
    """TabPFN's +2 s box for every agent row of a held-out episode, sized to 80% on validation
    episodes, exactly as in experiments/evidence.py Test 1 (same context rows, same sizing)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import evidence as EV
    from demo import apply_scale, boxes, calibrate, needed_scale
    tr, va, te = EV.splits(scene)
    ctx, va = EV.sample(tr, EV.TAB_CTX, 0), EV.sample(va, EV.N_VAL, 0)
    q = te[(te.episode == episode) & np.isclose(te.h, h)]
    m = EV.TabPFN().fit(EV.F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
    Q = m.predict_quantiles(np.concatenate([EV.F.X_of(va), EV.F.X_of(q)]), EV.QS)
    scales = calibrate(Q[:len(va)], va)
    Qc = apply_scale(Q[len(va):], q, scales)
    corners, mx, my, inside, _ = boxes(q, Qc)
    dt = 0.2
    out = [[int(a), int(round(t / dt)), *np.round(c.ravel(), 2).tolist(), round(float(tx), 2), round(float(ty), 2), int(i)]
           for a, t, c, tx, ty, i in zip(q.agent_id, q.t, corners, q.true_x, q.true_y, inside)]
    print(f"TabPFN boxes: {len(out)} predictions, {inside.mean():.0%} inside on this episode "
          f"(scale {scales[h]:.2f}, sized on {len(va)} validation rows)")
    vh = np.isclose(va.h.to_numpy(), h)
    need = needed_scale(Q[:len(va)][vh], va[["ddx", "ddy"]].to_numpy()[vh]) / scales[h]   # 1.0 = the box shown
    pred = dict(model="TabPFN", h=h, every=int(round(np.median(np.diff(np.unique(q.t))) / dt)), rows=out)
    pred["intro"] = intro_focus(q, corners, np.asarray(mx), np.asarray(my), inside, need, len(ctx))
    pred["compare"] = compare_boxes(EV, tr, ctx, va, q, pred["intro"], h)
    return pred


def compare_boxes(EV, tr, ctx, va, q, it, h):
    """The intro person's +2 s box from each model, sized on validation exactly as in evidence Test 1:
    constant velocity + fixed box, LightGBM (tuned, same 5,000 rows) and TabPFN."""
    from demo import apply_scale, box_area, boxes, calibrate
    row = q[(q.agent_id == it["agent"]) & np.isclose(q.t, it["k"] * 0.2)]
    Xv, Xr = EV.F.X_of(va), EV.F.X_of(row)
    models = {"cv_box": EV.M.ConstantBox().fit(EV.F.X_of(tr), tr[["ddx", "ddy"]].to_numpy()),
              "lgbm": EV.TunedLGBM(**EV.tune_lgbm(ctx, va)).fit(EV.F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()),
              "tabpfn": EV.TabPFN().fit(EV.F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())}
    out = {}
    for name, m in models.items():
        Q = m.predict_quantiles(np.concatenate([Xv, Xr]), EV.QS)
        Qr = apply_scale(Q[len(va):], row, calibrate(Q[:len(va)], va))
        c, mx, my, inside, _ = boxes(row, Qr)
        out[name] = dict(corners=np.round(c[0], 2).tolist(), median=[round(float(mx[0]), 2), round(float(my[0]), 2)],
                         area=round(float(box_area(Qr)[0]), 2), inside=bool(inside[0]))
    print("this person's +2 s box: " + ", ".join(f"{k} {v['area']} m2 ({'inside' if v['inside'] else 'missed'})" for k, v in out.items()))
    return out


def evidence_numbers(path="results/evidence.json"):
    """The street, ETH and Hotel results the outro shows (nothing recomputed here)."""
    p = Path(path)
    if not p.exists():
        return None
    E = json.loads(p.read_text())["scenes"]
    st = E["street"]
    t1 = {m: {k: st["test1"][m]["2"][k] for k in ("coverage", "area_m2", "error_m", "crowded_over_open")} for m in st["test1"]}
    t2 = {m: {b: c["err"] for b, c in cur.items()} for m, cur in st["test2"]["curves"].items()}
    t3 = {n: {m: [v[m]["mean"], v[m]["sd"]] for m in ("tabpfn", "lgbm")} for n, v in st["test3"].items()}
    real = {s: {m: {h: [E[s]["real_check"][m][h]["error_m"], E[s]["real_check"][m][h]["coverage"], E[s]["real_check"][m][h]["area_m2"]]
                    for h in ("1.2", "2")} for m in ("cv_box", "lgbm_all", "lgbm_same", "tabpfn")} for s in ("eth", "hotel") if s in E}
    return dict(t1=t1, t2=t2, t2_episodes=len(st["test2"]["episodes"]), t3=t3, real=real)


def intro_focus(q, corners, mx, my, inside, need, n_ctx):
    """Pick the person the intro explains: a pedestrian on the sidewalk at 7-12 s, moving, with
    neighbours, whose straight-line guess is clearly off and whose truth lands inside the patch."""
    cv_err = np.hypot(q.true_x - (q.px + 2 * q.vx), q.true_y - (q.py + 2 * q.vy)).to_numpy()
    ok = ((q.t >= 7) & (q.t <= 12) & (q.ext_type_pedestrian == 1) & (q.speed > 0.8) & (q.n0_dist < 3.0)
          & (np.abs(q.py) > 4.7)).to_numpy() & (inside == 1) & (cv_err > 0.5) & (cv_err < 2.0)
    i = int(np.flatnonzero(ok)[np.argmax(cv_err[ok])])
    r = q.iloc[i]
    row = lambda x: dict(id=int(x.agent_id), type=("car" if x.ext_type_car else "cyclist" if x.ext_type_cyclist else "pedestrian"),
                         speed=round(float(x.speed), 2), turn=round(float(np.degrees(x.turn_rate)), 1),
                         n=[round(float(x[f"n{k}_dist"]), 1) for k in range(3)], robot=round(float(x.robot_dist), 1),
                         obst=round(float(x.obst_dist), 1), crowd=int(x.density_3m))
    same_t = q[np.isclose(q.t, r.t) & (q.agent_id != r.agent_id)]
    same_t = same_t[(same_t.speed > 0.5) & (np.degrees(np.abs(same_t.turn_rate)) < 90)]    # rows that read plainly
    others = [row(x) for _, x in same_t.drop_duplicates("agent_id").sort_values("ext_type_car", ascending=False).head(3).iterrows()]
    return dict(agent=int(r.agent_id), k=int(round(r.t / 0.2)), row=row(r), others=others,
                corners=np.round(corners[i], 2).tolist(), median=[round(float(mx[i]), 2), round(float(my[i]), 2)],
                truth=[round(float(r.true_x), 2), round(float(r.true_y), 2)],
                cv=[round(float(r.px + 2 * r.vx), 2), round(float(r.py + 2 * r.vy), 2)], cv_err=round(float(cv_err[i]), 2),
                need=np.round(np.sort(need), 3).tolist(), n_ctx=int(n_ctx))


def sensor_track(tracks, episode, agent, k_end, dt=0.2, noise=0.1, seconds=3.0):
    """Noisy readings of one agent's last `seconds` and the core's Kalman filter run over them."""
    from foresight.core.features import _kalman_cv
    g = tracks[(tracks.episode == episode) & (tracks.agent_id == agent)].sort_values("t")
    g = g[np.round(g.t / dt).astype(int) <= k_end].tail(int(seconds / dt) * 2 + 1)
    z = g[["x", "y"]].to_numpy() + np.random.default_rng(7).normal(0, noise, (len(g), 2))
    p, v = _kalman_cv(z, dt, r=noise)
    keep = slice(len(g) - int(seconds / dt) - 1, None)              # filter warmed up on the first half
    k = np.round(g.t.to_numpy() / dt).astype(int)
    return dict(k=k[keep].tolist(), noisy=np.round(z[keep], 2).tolist(), smooth=np.round(p[keep], 2).tolist(),
                v=np.round(v[-1], 2).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/street")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true", help="open the page in Chrome")
    ap.add_argument("--predict", action="store_true", help="draw TabPFN's +2 s range (held-out episodes only)")
    a = ap.parse_args()
    tracks = pd.read_csv(Path(a.data) / "tracks.csv", low_memory=False)
    meta = json.loads((Path(a.data) / "meta.json").read_text(encoding="utf-8"))
    data = export(tracks, meta, a.episode)
    if a.predict:
        name = Path(a.data).name
        if not name.endswith("_test"):
            raise SystemExit("--predict needs a held-out folder (data/street_test or data/doorway_test)")
        data["pred"] = predictions(name[:-5] if name != "crowd_test" else "plaza", a.episode)
        it = data["pred"]["intro"]
        it["sensor"] = sensor_track(tracks, a.episode, it["agent"], it["k"])
        data["evidence"] = evidence_numbers()
        print(f"intro: agent {it['agent']} at {it['k'] * 0.2:.1f} s, straight-line guess off by {it['cv_err']} m")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, separators=(",", ":")))
    prefix = ("doorway3d" if data["scene"] == "doorway" else "street3d") + ("_pred" if a.predict else "")
    out = Path(a.out or f"results/{prefix}_ep{a.episode}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(data['agents'])} agents, {data['n']} frames)")
    if a.open:
        subprocess.run(["open", "-a", "Google Chrome", str(out)], check=False)


if __name__ == "__main__":
    main()
