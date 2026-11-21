"""Rebuild the working deck from media/ and results/. Run after every new figure or result.

    python tools/build_deck.py          -> deck/Foresight.pptx

Slides carry a title, one visual and at most a few words; explanations go in the speaker
notes. Missing figures become placeholder slides, so the deck always shows what is still to do.
This is the collect-everything deck; the 5-minute final is filtered from it.
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MEDIA, RESULTS, OUT = ROOT / "media", ROOT / "results", ROOT / "deck" / "Foresight.pptx"
INK, MUTED, ACCENT = RGBColor(0x1F, 0x23, 0x28), RGBColor(0x8B, 0x94, 0x9E), RGBColor(0xE8, 0x59, 0x0C)
W, H = Inches(13.333), Inches(7.5)


def _text(slide, text, left, top, width, height, size=20, bold=False, color=INK):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
        p.font.name = "Segoe UI"
    return tb


def slide(prs, title, notes="", kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])            # blank
    if kicker:
        _text(s, kicker.upper(), Inches(0.6), Inches(0.35), Inches(12), Inches(0.4), 12, True, ACCENT)
    _text(s, title, Inches(0.6), Inches(0.6), Inches(12.1), Inches(0.9), 30, True)
    s.notes_slide.notes_text_frame.text = notes
    return s


def image(s, path, top=Inches(1.6), max_w=Inches(12.1), max_h=Inches(5.6)):
    p = MEDIA / path
    if not p.exists():
        _text(s, f"[figure to come: media/{path}]", Inches(0.6), Inches(3.2), Inches(12), Inches(1), 18, color=MUTED)
        return
    pic = s.shapes.add_picture(str(p), 0, top)
    ratio = min(max_w / pic.width, max_h / pic.height)
    pic.width, pic.height = int(pic.width * ratio), int(pic.height * ratio)
    pic.left = int((W - pic.width) / 2)


def bullets(s, lines, top=Inches(1.8), size=22):
    _text(s, "\n".join(lines), Inches(0.8), top, Inches(11.7), Inches(5), size)


def results_table(s, ev):
    rows = [("Direction", "Model", "Rows seen", "Error +1.2 s", "Error +2 s", "Error +4.8 s", "80% range holds truth")]
    label = {"cv": "Const. velocity", "cv_band": "Const. velocity + band", "lgbm": "LightGBM (1,500)",
             "lgbm_all": "LightGBM (all)", "tabpfn": "TabPFN (1,500)"}
    for pair, r in ev["results"].items():
        a, b = (x.replace("eth", "ETH").replace("hotel", "Hotel") for x in pair.split("->"))
        for k in ("cv", "cv_band", "lgbm", "lgbm_all", "tabpfn"):
            v = r[k]
            rows.append((f"{a} → {b}", label[k], f"{v['2']['train_rows']:,}",
                         *(f"{v[h]['err_mean']:.2f} m" for h in ("1.2", "2", "4.8")),
                         "—" if k == "cv" else f"{100 * v['2']['cov80']:.0f}%"))
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(0.6), Inches(1.6), Inches(12.1), Inches(0.38) * len(rows))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = shape.table.cell(i, j); c.text = val
            para = c.text_frame.paragraphs[0]; para.font.size = Pt(12); para.font.name = "Segoe UI"
            para.font.bold = (i == 0) or (row[1].startswith("TabPFN"))


def _closed_table(s):
    import pandas as pd
    p = RESULTS / "closed.csv"
    if not p.exists():
        _text(s, "[results/closed.csv to come]", Inches(0.6), Inches(3.2), Inches(12), Inches(1), 18, color=MUTED)
        return
    import sys
    sys.path.insert(0, str(ROOT))
    from run_closed import load_closed
    df = load_closed(p)
    df = df[(df.sim == "crowd") & (df.box_q == 0.05)]
    g = df.groupby("planner").agg(n=("seed", "nunique"), reached=("reached", "mean"), t=("time_to_goal", "mean"),
                                  coll=("collided", "mean"), gap=("min_gap", "mean"), freeze=("freeze_rate", "mean"))
    order = [k for k in ("cv", "cv_band", "lgbm", "tabpfn") if k in g.index]
    label = {"cv": "Const. velocity", "cv_band": "Const. velocity + band", "lgbm": "LightGBM", "tabpfn": "TabPFN"}
    rows = [("Planner", "Crowds", "Reached goal", "Time to goal", "Runs with a collision", "Closest gap (mean)", "Frozen")]
    for k in order:
        r = g.loc[k]
        rows.append((label[k], f"{int(r.n)}", f"{100 * r.reached:.0f}%", f"{r.t:.1f} s", f"{100 * r.coll:.0f}%",
                     f"{r.gap:.2f} m", f"{100 * r.freeze:.0f}%"))
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(0.6), Inches(1.8), Inches(12.1), Inches(0.45) * len(rows))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = shape.table.cell(i, j); c.text = val
            para = c.text_frame.paragraphs[0]; para.font.size = Pt(14); para.font.name = "Segoe UI"
            para.font.bold = (i == 0) or row[0] == "TabPFN"


def build():
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H
    ev = json.loads((RESULTS / "eval.json").read_text()) if (RESULTS / "eval.json").exists() else None

    s = slide(prs, "Foresight", kicker="HackLab Berlin · Prior Labs track",
              notes="One prediction-and-planning layer for any robot among people or cars. TabPFN predicts where each "
                    "nearby agent will be in the next seconds, as a range; the robot keeps clear of the ranges.")
    _text(s, "Where will everyone be in 2 seconds, and how sure are we?", Inches(0.6), Inches(1.6), Inches(12), Inches(1), 24, color=MUTED)
    _text(s, "Darshak Panseriya · Nooshin Pourkamali", Inches(0.6), Inches(6.4), Inches(12), Inches(0.5), 16, color=MUTED)

    s = slide(prs, "A robot in public needs the next 2 seconds", kicker="Problem",
              notes="Robots among people (delivery robots, humanoids, cars) must plan around where others will be, not "
                    "where they are. A single guess is not enough: the planner needs to know how sure the guess is.")
    bullets(s, ["Where will each person be?", "How sure is that?", "Which move stays clear?"])

    s = slide(prs, "Scene → table → TabPFN → ranges → robot", kicker="Approach",
              notes="Each agent at each moment becomes a table row (its own motion, nearest neighbours, robot, obstacles). "
                    "Target: deviation from constant velocity after h seconds, in the agent's own frame. TabPFN returns "
                    "quantiles: a range, not just a point. No training: it reads labelled example rows as context.")
    bullets(s, ["1 row = 1 agent, 1 moment, 1 horizon", "Target: deviation from constant velocity",
                "TabPFN: quantiles from example rows, no training", "Planner: pick the move clear of the ranges"])

    s = slide(prs, "Real pedestrians: ETH and Hotel (Zurich)", kicker="Data",
              notes="ETH Walking Pedestrians dataset (Pellegrini et al. 2009), via OpenTraj. Positions in metres every "
                    "0.4 s. ETH: 360 people; Hotel: 390 people. We take context rows from one scene and test on the "
                    "other, so the model never saw the test place. Plus a stand-in simulated crowd with a robot.")
    bullets(s, ["750 people, positions every 0.4 s", "Context from one place, test on the other",
                "Plus a simulated crowd with a robot"])

    s = slide(prs, "TabPFN on real people it never saw", kicker="Demo",
              notes="Hotel scene; TabPFN saw 1,500 rows from the ETH scene only. Orange box: 80% range for the position "
                    "2 s ahead. Black x: where the person actually was. Over this episode the box held the truth 94% "
                    "of the time, median error 0.26 m.")
    image(s, "demo_hotel_tabpfn.png")

    s = slide(prs, "Same people, four predictors", kicker="Comparison",
              notes="Same moment, same people. Constant velocity with a fixed band: honest but wide. LightGBM: narrow "
                    "boxes that miss more often. TabPFN: narrower than the band, holds the truth as often as promised.")
    image(s, "compare_hotel.png")

    s = slide(prs, "Error by horizon", kicker="Results",
              notes="Mean distance between predicted (median) and actual position. Both directions: context from one "
                    "scene, test on the other. LightGBM (all rows) is its best case: every training row of the source scene.")
    image(s, "fig_error_by_horizon.png")

    s = slide(prs, "Are the ranges honest?", kicker="Results",
              notes="For ranges promising 20 to 90% coverage, how often the actual position fell inside. On the dotted "
                    "line = honest. Below = overconfident (dangerous for a robot). Above = too cautious (slows it down).")
    image(s, "fig_calibration.png")

    s = slide(prs, "How many examples does it need?", kicker="Results",
              notes="Error at +2 s against the number of example rows, ETH to Hotel (one seed). TabPFN leads at every size. "
                    "Both models are best at 250-500 rows from the other place and get worse with more: see next slide.")
    image(s, "fig_data_efficiency.png")

    s = slide(prs, "Which rows, not how many", kicker="Finding",
              notes="LightGBM, 5 seeds, +2 s. Rows from another place help up to ~500, then hurt (0.32 to 0.39 m). Rows "
                    "from the same place keep helping (0.37 to 0.28 m). The test sets differ, so compare shapes. This is why "
                    "the next step chooses context rows per moment: from this scene and similar situations.")
    image(s, "fig_rows_place.png")

    s = slide(prs, "Robot crossing a crowd: two planners", kicker="Closed loop",
              notes="Same simulated crowd (Nooshin's social-force simulator), same start and goal. Left: constant-velocity "
                    "predictions with a safety margin. Right: TabPFN 80% ranges. The robot picks, every 0.4 s, the move "
                    "that gets closest to its goal while keeping clear of the predicted ranges at +1 s and +2 s.")
    image(s, "closed_crowd_1001_cv_vs_tabpfn.png")

    s = slide(prs, "Fast, safe, or both?", kicker="Closed loop",
              notes="Six crowds per planner (seeds 1000-1005), the same for every planner. Collision = robot touching a "
                    "person while moving. Freeze = share of decisions where the robot stood still.")
    _closed_table(s)

    s = slide(prs, "Any agent: people, cars, robots", kicker="Scenarios",
              notes="Every scenario writes the same track format. Done: simulated plaza, real ETH/Hotel pedestrians. Next: "
                    "mixed street (cars in lanes, pedestrians on pavements, a delivery robot crossing). Pitch only: "
                    "warehouse floor robots, drone swarms.")
    bullets(s, ["Plaza crowd + robot (simulated)", "Real pedestrians, Zurich", "Next: street with cars, people, delivery robot",
                "Same layer: warehouse robots, drones"])

    s = slide(prs, "Numbers", kicker="Results",
              notes="Mean position error at three horizons; coverage of the 80% range at +2 s. TabPFN and LightGBM "
                    "(1,500) see the same 1,500 random rows. LightGBM (all) and the band use every training row.")
    if ev:
        results_table(s, ev)
    else:
        _text(s, "[results/eval.json to come]", Inches(0.6), Inches(3.2), Inches(12), Inches(1), 18, color=MUTED)

    s = slide(prs, "Baselines, built fairly", kicker="Method",
              notes="Constant velocity: last filtered velocity, held. Constant velocity + band: fixed range from training "
                    "residual quantiles per horizon. LightGBM: one quantile-regression model per quantile per axis, 300 "
                    "trees, learning rate 0.05, 31 leaves, min 10 rows per leaf, default otherwise, no tuning; same "
                    "features and rows as TabPFN, plus a version on all rows. Split by place: no test person in context.")
    bullets(s, ["Constant velocity (± fixed band)", "LightGBM quantile regression, same features",
                "Same rows as TabPFN, and all rows", "Tested on a place none of them saw"])

    s = slide(prs, "Next", kicker="Plan",
              notes="R2: per-moment context selection (recent rows + similar situations) and adaptation after a "
                    "behaviour change. R3: robot planner choosing moves clear of the ranges, against baselines. R4: same "
                    "layer on highway-env cars.")
    bullets(s, ["Pick the right examples per moment", "Robot chooses its path from the ranges", "Same layer drives a car"])

    OUT.parent.mkdir(exist_ok=True); prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
