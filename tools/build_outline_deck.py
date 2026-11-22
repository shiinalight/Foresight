"""Skeleton finalist deck from docs/DECK_OUTLINE.md: one slide per outline slide, headline as title,
the outline's pictures placed per its layout line, the spoken text in the speaker notes.
Plain on purpose: the team finishes the deck by hand in PowerPoint.

    python tools/build_outline_deck.py          # writes deck/Foresight_finalist.pptx
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deck/Foresight_finalist.pptx"
W, H = 13.333, 7.5
M = 0.5
INK, MUTED, ACCENT = RGBColor(0x1C, 0x22, 0x26), RGBColor(0x5A, 0x60, 0x66), RGBColor(0xC2, 0x4E, 0x1C)
FONT = "Calibri"

S = "media/stills/"
D = "media/diagrams/"
G = "media/demo/"

# Each slide: tag (shown small, top left), title, rows of pictures (one row: list of paths),
# a bold line under the pictures, a to-do line, and the speaker notes.
SLIDES = [
    dict(tag="1 · Title · 15 s · Darshak", title="Foresight", cover=S + "3d_range_every_agent.jpg",
         sub="Where will everyone be in two seconds, and how sure are we?\nNooshin Pourkamali · Darshak Panseriya · Prior Labs track",
         notes="Say: A delivery robot shares the street with people, cars and bikes. Foresight tells it where they'll be next, and how much to trust that."),
    dict(tag="2 · The problem · 30 s · Darshak", title="Robots guess with a straight line and a fixed margin",
         pics=[S + "3d_problem_093m.jpg", D + "01_problem.png"],
         line="Walking straight is off by 0.93 m after 2 s in a crowd; 0.25–0.43 m on real Zurich pedestrians",
         notes="Say: today's options are to assume everyone walks straight, add one margin, or stop when someone is close. Stock ROS 2 Nav2 plans around where people are now. Learned forecasters do better, but once trained they are frozen. Too narrow a margin and the robot hits someone; too wide and it freezes."),
    dict(tag="3 · The idea · 30 s · Darshak", title="The robot's knowledge is a table",
         pics=[D + "03_row.png", D + "05_live_context.png"], weights=[1, 1.7],
         line="No training step. Adapting = swapping rows.",
         notes="Say: each person, at each moment, is one row: their motion, neighbours, the robot, obstacles. TabPFN reads a few thousand example rows and answers in one pass, with no training. To adapt to a new place, a new behaviour or a new body, change the rows.\nDetail: frozen bank of 2,000 rows plus up to 250 rows of the current scene, refit every 2 s; 5–95% range per person at +1 and +2 s; weights never change on the robot."),
    dict(tag="4 · How it reads one person · 30 s · Nooshin", title="From sensor dots to a range, in four steps",
         pics=[S + "3d_step1_watching.jpg", S + "3d_step2_row.jpg", S + "3d_step4_tabpfn.jpg", S + "3d_step5_sizing.jpg"],
         labels=["1 Watch", "2 Row", "3 TabPFN", "4 Size"],
         todo="Option: play this part of the 3D video (about 40 s) instead of the stills",
         notes="Say: a filter smooths the noisy track; the moment becomes a row; TabPFN predicts the correction to walking straight as a 5–95% range; the range is sized so the truth lands inside as often as promised."),
    dict(tag="5 · Watch it on a street · 35 s · Nooshin", title="A street run the model has never seen",
         pics=[S + "3d_street_sudden_stop.jpg", S + "3d_street_zebra_brake.jpg", S + "3d_street_rush_hour.jpg"],
         labels=["Stops suddenly", "Car brakes for the zebra", "Rush hour"],
         todo="Insert the 3D street clip here: media/demo_full.mp4 on nooshin/dev, about 1:03–1:43 (no sound). Caption: simulated street; scripted robot route; ranges from TabPFN",
         notes="Say: someone stops dead, a car brakes for the zebra, a jaywalker leaves every path, rush hour changes the crowd. Every agent gets a range.\nThe street's ranges come from a fixed example set: don't say it adapts live here. Slide 6 shows adaptation."),
    dict(tag="6 · Proof 1 · 35 s · Darshak", title="When the crowd changed, reading the scene kept more people inside their ranges",
         pics=[G + "seed3013.gif", "media/fig_t5.png"],
         line="59% vs 41% inside the 2 s range (+18 pts) · +10 from the scene alone · ahead in 3 of 3 crowds",
         notes="Say: in our robot loop the crowd turns erratic five seconds in. The TabPFN that reads the last seconds kept 59% of people inside their 2 s range; LightGBM kept 41%. Ten of those points come from reading the scene alone. Ahead in all three crowds (+6.6, +18.0, +19.7 pts).\nTarget is 81%: closer, not there. Use the right panel of fig_t5 (or media/demo/calibration_seed3013_strip.png). If asked whether LightGBM had the scene rows: backup B1."),
    dict(tag="7 · Proof 2 · 30 s · Darshak", title="The same core predicts highway traffic",
         pics=["media/demo/highway.gif"],
         line="1,500 rows match LightGBM on 100,440 · 86% vs 69–82% inside · +14 pts after the switch",
         notes="Say: same features and models, a small adapter for cars. 1,500 example rows match LightGBM trained on 67 times more, with more honest ranges. When drivers turn aggressive, reading the scene adds 14 points.\nSimulated traffic (highway-env), offline. Say 'matches', never 'beats'. Each body needs its own example bank."),
    dict(tag="8 · Proof 3 · 30 s · Darshak", title="On five real scenes, TabPFN gets more from each example",
         pics=["media/demo/scenes5.gif", "media/fig_loso5.png"], weights=[1, 1.4],
         line="Beats the fixed margin in all 5 scenes · beats LightGBM on equal rows in 4 of 5 · tie at ~15× the rows",
         notes="Say: ETH, Hotel, Zara 1 and 2, Univ: train on four, test on the fifth. TabPFN beats the fixed-margin guess from slide 2 in all five scenes. On the same rows it beats LightGBM in four scenes and ties on Hotel. LightGBM needs about 15 times the data to catch up.\nAgainst constant velocity plus a fixed band: −0.365 [−0.405, −0.328] pooled (forecast quality only). Interval score −0.130 pooled at equal rows; −0.008 (tie) at 15x. Zara1 and Zara2 share one camera."),
    dict(tag="9b · Optional · 20 s", title="A second TabPFN can pick the robot's move",
         pics=[G + "policy_seed3403.gif", D + "02_pipeline.png"],
         todo="Circle the three TabPFN stages on the pipeline diagram. Cut this slide first if the talk runs long",
         notes="Say: ranges, scene adaptation and the move are all TabPFN calls. On 20 new crowds the TabPFN policy drove like the planner it learned from on average.\nTracker, calibration offset and the hard safety check stay classical. Collision interval is wide at 20 crowds: don't say 'as safe'."),
    dict(tag="9 · Where it breaks · 30 s · Darshak", title="Better forecasts didn't make our robot faster",
         pics=["media/fig_t5.png"], half=True,
         bullets=["3–4 s slower than standard planners at equal safety", "New place: holds on two real scenes, a tie on five",
                  "Each body needs its own example bank", "Robot loop and cars are simulated"],
         notes="Say: the planner turns the range into a yes/no box. Its caution setting set the speed, not the forecast. That planner is the next thing to build.\nUse the left panel of fig_t5 (time against closest gap). If asked how the forecast beats a fixed margin while the constant-velocity planner beat the robot: the ranges are better in every real scene, but this planner turns each range into a yes/no box, so its caution setting sets the pace. Backup B7 for 'why slower'."),
    dict(tag="10 · What it's for · 35 s · Darshak", title="A forecasting layer you re-target by swapping rows",
         pics=[D + "12_deployment.png"],
         line="Few examples or a sudden change: TabPFN · plenty of familiar data: LightGBM · 71 ms on an L4",
         todo="Update the diagram: '2,000-row bank + 250 scene rows, every 0.4 s' and 'safety check: 0.4 s straight-line veto' (the takeover is not built)",
         notes="Say: Foresight sits under whatever planner a team already runs. New site, new crowd behaviour, new vehicle: add rows, no training run. Next: a planner that uses the full range, a closed loop for cars, real robot logs.\nClose: every number in this talk comes from a script in the repo."),
    dict(tag="Backup", title="Backup slides", divider=True, notes="For questions only."),
    dict(tag="B1 · Is proof 1 fair?", title="LightGBM never saw the scene. Is proof 1 fair?", pics=["media/fig_live_variants_cp.png"],
         line="Scene rows add 10 pts; LightGBM refit on the same scene still covers ~12 pts less (offline)",
         notes="Offline, simulated, after the switch: +12.3 pts 80% coverage [+7.7, +17.0], pinball −0.015, error level (results/live_variants.csv, post-hoc)."),
    dict(tag="B2 · Tuned until it worked?", title="Did you tune until it worked?",
         bullets=["Every pass rule was committed before its run (pre-registration with amendments)", "Paired crowds, bootstrap intervals, Holm correction",
                  "Claim C failed and claim A narrowed under those rules", "Failures stay in the README"],
         notes="Commit hashes for each rule are in RESULTS.md and docs/DECISIONS.md."),
    dict(tag="B3 · Data", title="What data?", pics=["media/fig_f4_data_cards.png"],
         line="Five real scenes (ETH, Hotel, Zara 1–2, Univ) · three simulated crowds · highway-env traffic",
         notes="Real data: Pellegrini 2009 (ETH, Hotel), Lerner 2007 (UCY), read through OpenTraj. Annotated every 0.4 s, the step we replan on."),
    dict(tag="B4 · How much data", title="How much data does TabPFN need?", pics=["media/fig_f4_context_size.png"],
         line="2,000 rows tie LightGBM on 20,000", notes="Crossing crowd, +2 s, f4 features, 3 seeds (results/context_size.csv)."),
    dict(tag="B5 · New place", title="Ranges in a new place?", pics=["media/fig_f4_claim_a.png"],
         line="Better than LightGBM on two scenes; a tie on five when LightGBM gets all rows", notes="Two scenes: 75% vs 58% box coverage (Hotel to ETH). Five scenes: −0.008, a tie."),
    dict(tag="B6 · Turns", title="Can it predict turns?", pics=["media/fig_intent_layer.png"],
         line="Wider ranges for people who turn (46% → 63% inside, real data); it does not call turns",
         notes="Intent chain: TabPFN classifier, then per-manoeuvre ranges. Offline only; turn recall is 0."),
    dict(tag="B7 · Why slower", title="Why was the robot slower?", pics=["media/fig_f4_oracle_gate.png"],
         line="Same caution setting, same speed for every forecaster; even true futures gain under 2 s",
         notes="Split − frozen TabPFN −0.12 s [−1.35, +1.25] at q 0.15. The box veto throws away the range's shape."),
    dict(tag="B8 · Speed, features, references", title="Fast enough? Feature engineering? Why these choices?",
         bullets=["Speed: 71 ms per replan on an NVIDIA L4, 318 ms scene rebuild; Apple M3 Pro 287–321 ms; fits the 0.4 s replan",
                  "Feature engineering: 21 hand-made properties (time to closest approach, threat, companion): no gain",
                  "Constant velocity baseline: Schöller et al. 2020 · TabPFN: Hollmann et al. 2025 (Nature)",
                  "Data: Pellegrini 2009, Lerner 2007 · calibration: Gibbs & Candès 2021 · full list: docs/VIDEO_BRIEF.md section 4"],
         notes="10 Hz is not claimed: it needs a background scene rebuild that is designed, not built."),
    dict(tag="B9 · One decision", title="One decision in detail", pics=["docs/walkthrough/s5_person_box.png", "docs/walkthrough/s8_planner.png"],
         line="One real moment, every stage, with the actual values (docs/walkthrough.html)", notes="Seed 6001, t = 7.2 s, person 11."),
]


def text(slide, x, y, w, h, s, size, bold=False, color=INK, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, part in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = part
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
        r.font.color.rgb = color
    return tb


def picture(slide, rel, x, y, w, h, label=None):
    path = ROOT / rel
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    text(slide, x, y + h + 0.02, w, 0.3, label or rel, 10, color=MUTED)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    for sp in SLIDES:
        s = prs.slides.add_slide(blank)
        s.notes_slide.notes_text_frame.text = sp.get("notes", "")
        if sp.get("cover"):
            s.shapes.add_picture(str(ROOT / sp["cover"]), 0, 0, Inches(W), Inches(H))
            band = s.shapes.add_shape(1, 0, Inches(4.45), Inches(W), Inches(H - 4.45))   # 1 = rectangle
            band.fill.solid()
            band.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xF8)
            band.line.fill.background()
            text(s, M, 4.6, W - 2 * M, 0.35, sp["tag"], 14, bold=True, color=ACCENT)
            text(s, M, 5.0, W - 2 * M, 0.9, sp["title"], 54, bold=True)
            text(s, M, 5.9, W - 2 * M, 0.9, sp["sub"], 20, color=INK)
            continue
        text(s, M, 0.3, W - 2 * M, 0.3, sp["tag"], 12, bold=True, color=ACCENT)
        if sp.get("divider"):
            text(s, M, 3.0, W - 2 * M, 1.0, sp["title"], 44, bold=True)
            continue
        text(s, M, 0.6, W - 2 * M, 1.0, sp["title"], 30, bold=True)
        top, bottom = 1.75, 6.15
        if sp.get("line"):
            text(s, M, 6.45, W - 2 * M, 0.5, sp["line"], 16, bold=True)
        if sp.get("todo"):
            text(s, M, 6.98, W - 2 * M, 0.45, "TO DO: " + sp["todo"], 10, color=ACCENT)
        pics = sp.get("pics", [])
        bullets = sp.get("bullets")
        area_w = (W - 2 * M) / 2 - 0.2 if (bullets and pics) else W - 2 * M
        if pics:
            n, gap = len(pics), 0.3
            wts = sp.get("weights", [1] * n)
            x = M
            for i, rel in enumerate(pics):
                pw = (area_w - gap * (n - 1)) * wts[i] / sum(wts)
                lab = sp["labels"][i] + "  ·  " + rel if sp.get("labels") else rel
                picture(s, rel, x, top, pw, bottom - top, label=lab)
                x += pw + gap
        if bullets:
            bx = M + area_w + 0.4 if pics else M
            bw = W - M - bx
            tb = text(s, bx, top + 0.3, bw, bottom - top - 0.3, "", 20)
            tf = tb.text_frame
            for i, b in enumerate(bullets):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                r = p.runs[0] if (i == 0 and p.runs) else p.add_run()
                r.text = "–  " + b
                r.font.size, r.font.name = Pt(20), FONT
                r.font.color.rgb = INK
                p.space_after = Pt(12)
    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
