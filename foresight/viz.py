"""Shared drawing: scene with predicted ranges, and chart styling. Minimal text in figures;
explanations live in RESULTS.md and the deck notes."""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.transforms import Affine2D

from foresight.core import features as F

# validated categorical palette (dataviz validator, light mode)
C_PERSON, C_PRED, C_ROBOT = "#2a6fdb", "#e8590c", "#7a3fb5"
C_INK, C_MUTED, C_OBST, C_GRID = "#1f2328", "#8b949e", "#c9ced6", "#eceef1"
# model colours for charts: TabPFN keeps the prediction orange; constant velocity is neutral grey
MODEL_COLOR = {"tabpfn": "#e8590c", "lgbm": "#2a6fdb", "cv_band": "#7a3fb5", "cv": "#8b949e"}
MODEL_LABEL = {"tabpfn": "TabPFN", "lgbm": "LightGBM", "cv_band": "Const. velocity + band", "cv": "Const. velocity"}
QS_BOX = (0.05, 0.5, 0.95)      # per-axis 5-95% -> box holds the truth ~80% of the time


def predict_rows(model, q, qs=QS_BOX):
    """Add median world position, box bounds, inside flag and error to rows q.
    qs = (low, 0.5, high) per axis; the default 5-95% gives a box holding the truth ~80% of the time."""
    q = q.copy()
    Q = model.predict_quantiles(F.X_of(q), qs)
    q["mx"], q["my"] = F.to_world(q, Q[:, 0, 1], Q[:, 1, 1])
    q["lo_a"], q["hi_a"], q["lo_c"], q["hi_c"] = Q[:, 0, 0], Q[:, 0, 2], Q[:, 1, 0], Q[:, 1, 2]
    Y = q[["ddx", "ddy"]].to_numpy()
    q["inside"] = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 2])).all(axis=1)
    q["err"] = np.hypot(q.mx - q.true_x, q.my - q.true_y)
    return q


def style_axes(ax, bounds=None, equal=True):
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
    if equal:
        ax.set_aspect("equal")
    ax.set_facecolor("white"); ax.grid(color=C_GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def draw_scene(ax, tr_ep, obst_ep, rows_t, t, bounds, axis_labels=True):
    """One moment: people, last-2 s paths, robot, obstacles, predicted boxes, true positions."""
    ax.clear(); style_axes(ax, bounds)
    if axis_labels:
        ax.set_xlabel("x (m)", color=C_MUTED, fontsize=9); ax.set_ylabel("y (m)", color=C_MUTED, fontsize=9)
    for _, o in obst_ep.iterrows():
        ax.add_patch(Circle((o.x, o.y), o.r, color=C_OBST, zorder=1))
    past = tr_ep[(tr_ep.t <= t + 1e-6) & (tr_ep.t >= t - 2.0 - 1e-6)]
    for _, g in past.groupby("agent_id"):
        ax.plot(g.x, g.y, color=C_ROBOT if g.is_ego.iloc[0] else C_MUTED, lw=1.0, alpha=0.8, zorder=2)
    for _, r in tr_ep[np.isclose(tr_ep.t, t)].iterrows():
        if r.is_ego:
            ax.plot(r.x, r.y, marker="s", color=C_ROBOT, ms=10, zorder=6)
        else:
            ax.add_patch(Circle((r.x, r.y), r.radius, color=C_PERSON, zorder=5))
    for _, r in rows_t.iterrows():
        cvx, cvy = r.px + r.vx * r.h, r.py + r.vy * r.h
        box = Rectangle((r.lo_a, r.lo_c), max(r.hi_a - r.lo_a, 0.05), max(r.hi_c - r.lo_c, 0.05),
                        fill=False, color=C_PRED, lw=1.4, zorder=4)
        box.set_transform(Affine2D().rotate(r.theta).translate(cvx, cvy) + ax.transData)
        ax.add_patch(box)
        ax.plot([r.px, r.mx], [r.py, r.my], color=C_PRED, lw=1, zorder=4)
        ax.plot(r.mx, r.my, "o", color=C_PRED, ms=3, zorder=4)
        ax.plot(r.true_x, r.true_y, "x", color=C_INK, ms=6, mew=1.4, zorder=7)


def scene_legend(h, has_robot=False, has_obstacles=False):
    items = [
        Line2D([], [], marker="o", ls="", color=C_PERSON, ms=8, label="Person"),
        Line2D([], [], color=C_MUTED, lw=1.2, label="Path, last 2 s"),
        Line2D([], [], marker="o", ls="-", color=C_PRED, ms=4, lw=1, label=f"Predicted, +{h:g} s"),
        Patch(facecolor="none", edgecolor=C_PRED, lw=1.5, label="80% range"),
        Line2D([], [], marker="x", ls="", color=C_INK, ms=7, mew=1.6, label=f"Actual, +{h:g} s"),
    ]
    if has_robot:
        items.append(Line2D([], [], marker="s", ls="", color=C_ROBOT, ms=9, label="Robot"))
    if has_obstacles:
        items.append(Patch(facecolor=C_OBST, label="Obstacle"))
    return items
