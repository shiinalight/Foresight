"""Rebuild the working deck from media/ and results/. Run after every new figure or result.

    python tools/build_deck.py          -> deck/Foresight.pptx

Slides carry a title, one visual and at most a few words; explanations go in the speaker
notes. Missing figures become placeholder slides, so the deck always shows what is still to do.
Final deck on the f4 verdicts (RESULTS.md, Sun 22 Nov): what works (claims A, B, data efficiency, intent),
what did not (claim C, closed-loop speed) and why.
"""
from __future__ import annotations

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


def strip(s, paths, top=Inches(1.7), max_h=Inches(4.9), gap=Inches(0.15)):
    """Several images side by side, equal height, centred; missing ones become placeholders."""
    pics = []
    for path in paths:
        p = MEDIA / path if not str(path).startswith("docs/") else ROOT / path
        if not p.exists():
            _text(s, f"[figure to come: {path}]", Inches(0.6), Inches(3.2), Inches(12), Inches(1), 18, color=MUTED)
            return
        pics.append(s.shapes.add_picture(str(p), 0, top))
    for pic in pics:
        r = max_h / pic.height; pic.width, pic.height = int(pic.width * r), int(pic.height * r)
    total = sum(p.width for p in pics) + gap * (len(pics) - 1)
    if total > Inches(12.3):                                   # shrink to fit the width
        r = Inches(12.3) / total
        for pic in pics:
            pic.width, pic.height = int(pic.width * r), int(pic.height * r)
        total = sum(p.width for p in pics) + gap * (len(pics) - 1)
    x = int((W - total) / 2)
    for pic in pics:
        pic.left = x; x += pic.width + gap


def caption(s, text, top=Inches(6.85)):
    _text(s, text, Inches(0.6), top, Inches(12.1), Inches(0.5), 14, color=MUTED)


def build():
    """The final deck on the f4 verdicts (RESULTS.md, Sun 22 Nov). Numbers in notes and captions come from RESULTS.md."""
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H

    s = slide(prs, "Foresight", kicker="HackLab Berlin · Prior Labs track",
              notes="One prediction layer for a robot among people. TabPFN reads a table of what people just did and "
                    "returns, for each person, a range for where they will be in one and two seconds. The robot keeps "
                    "clear of the ranges. Darshak Panseriya, Nooshin Pourkamali.")
    _text(s, "Where will everyone be in two seconds, and how sure are we?", Inches(0.6), Inches(1.4), Inches(12), Inches(0.8), 22, color=MUTED)
    image(s, "../docs/walkthrough/s6_all_boxes.png", top=Inches(2.2), max_h=Inches(4.4))
    caption(s, "Darshak Panseriya · Nooshin Pourkamali")

    s = slide(prs, "A robot needs the next two seconds, and how sure it is", kicker="Problem",
              notes="Walking straight is a strong baseline on real people (Schöller et al. 2020): constant velocity is "
                    "off by 0.37 m (ETH) and 0.25 m (Hotel) at +2 s. The hard part is knowing how wrong it might be, "
                    "and that is hardest in a place the robot has never seen, where an overconfident model is dangerous.")
    image(s, "diagrams/01_problem.png")

    s = slide(prs, "Scene → table → TabPFN → ranges → robot", kicker="Approach",
              notes="One row = one person, one moment, one look-ahead: their own motion, three nearest neighbours, the "
                    "robot, in the person's own frame. The label is how far they end up from walking straight. TabPFN "
                    "reads labelled rows as examples and returns quantiles in one pass, no training.")
    image(s, "diagrams/02_pipeline.png")

    s = slide(prs, "One real decision, stage by stage", kicker="How it works",
              notes="Real values at one moment, picked by a fixed rule (docs/walkthrough.html, f4, seed 6001): the "
                    "example rows TabPFN read, one person's range, and the planner vetoing moves that enter a range.")
    strip(s, ["docs/walkthrough/s4_context.png", "docs/walkthrough/s5_person_box.png", "docs/walkthrough/s8_planner.png"])
    caption(s, "Examples it read  ·  one person's range  ·  the move the planner keeps")

    s = slide(prs, "Tested so it can fail", kicker="Method",
              notes="Every pass condition was written before its run. A leak was found and fixed (D-20). Our constant "
                    "velocity matches the published one (ETH 1.12 vs 1.15 m, Hotel 0.46 vs 0.51 m at 4.8 s, D-26). The "
                    "plaza could not separate planners (T1), so the crowd was chosen by a rule from baselines alone (T2). "
                    "The oracle gate failed (D-33), so the closed loop is a stress test.")
    image(s, "diagrams/11_tests.png")

    s = slide(prs, "Honest ranges in a place it never saw", kicker="Result · claim A",
              notes="Real Zurich pedestrians, context from one scene, test on the other, 5 context samples, paired person "
                    "bootstrap. Hotel → ETH: TabPFN's 5-95% box holds the truth 74.8% (LightGBM on all rows 57.7%); "
                    "interval score −0.303 [−0.425, −0.184] against LightGBM. ETH → Hotel: 86.4% vs 74.9%, interval "
                    "score level with LightGBM on all rows. Direction-dependent; in the hard direction TabPFN still "
                    "under-covers (75% against about 81%). Point error level with LightGBM, slightly better than tuned CV. On five scenes (leave one out, "
                    "docs/LOSO5.md): edge strongest when data is scarce; ties a rival trained on 15x more varied data "
                    "(pooled −0.008 [−0.023, +0.007]); on the same 1,500 rows it beats LightGBM (−0.130).")
    image(s, "fig_f4_claim_a.png")

    s = slide(prs, "Few rows are enough", kicker="Result · data efficiency",
              notes="TabPFN on 2,000 rows is about level with LightGBM on 20,000 (results/context_size.csv, crowd_cross, "
                    "f4). Retrieved (similar) rows did not beat random rows at equal size.")
    image(s, "fig_f4_context_size.png")

    s = slide(prs, "TabPFN at two chained layers", kicker="Result · intent",
              notes="A TabPFN classifier gives manoeuvre probabilities (straight, stop, turn left or right); per-manoeuvre "
                    "TabPFN range models give ranges, mixed by those probabilities. Real Hotel → ETH: on people who turn, "
                    "the interval score improves by −0.263 [−0.481, −0.075] (73 rows, 37 people) and coverage rises from "
                    "45.7% to 62.6%; overall level. It does NOT forecast turns: turn recall is 0. The gain is calibrated "
                    "widening from the mixture. Say 'better-calibrated ranges for people who turn'. In the simulator "
                    "intent is random by construction and the chain is not adopted (D-34). Offline only.")
    image(s, "fig_intent_layer.png")

    s = slide(prs, "Same core, another form factor", kicker="Result · highway cars",
              notes="The core, unchanged, on highway-env cars (4 lanes, 20 IDM/MOBIL cars; offline, simulated; "
                    "docs/HIGHWAY.md, rules fixed before any TabPFN run). H1 passed: TabPFN on 1,500 rows holds 85.6% "
                    "with interval score 3.57, against LightGBM on 100,440 rows at 82.1% / 4.06 (−0.49 [−0.66, −0.34]) "
                    "and constant velocity plus band at 82.9% / 7.78. H2 passed: after traffic turns aggressive, a style "
                    "the bank never saw, split beats frozen (coverage 76.7% vs 62.5%, interval score −0.98 [−1.41, "
                    "−0.56]); about 80% of the gain is in the first 6 s after the switch. A pedestrian bank on cars gives "
                    "~50 m boxes, so each embodiment needs its own bank (D-35). No closed loop on cars.")
    image(s, "fig_highway.png", max_h=Inches(5.1))
    caption(s, "Cars: TabPFN 85.6% coverage, interval score 3.57 vs LightGBM (67x more rows) 4.06 · adapts after traffic turns aggressive")

    s = slide(prs, "Adapts in the loop when behaviour changes", kicker="Result · claim B",
              notes="20 fresh simulated crowds that turn erratic at 5 s; erratic was never in the bank. After the switch, "
                    "reading this scene's last completed rows (split) keeps the box holding the truth 59.3% of the time, "
                    "against 49.3% for frozen TabPFN (+10.0 points [+8.4, +11.6]) and 41.3% for LightGBM; interval "
                    "score 5.03 vs 6.34 (−1.31 [−1.55, −1.08]); error −0.057 m. Before the switch it costs a little. "
                    "No model reaches the 81% target. Against LightGBM it holds in three crowds (open plaza +6.6, crossing "
                    "+18.0, dense +19.7 points; GENERALISATION.md). Simulation; real data: a small gain (RESULTS.md).")
    image(s, "fig_t5.png", max_h=Inches(5.15))
    caption(s, "Right panel: box holds the true position at +2 s, before and after the switch (20 paired crowds)")

    s = slide(prs, "Calibrated, but slower", kicker="Closed loop · claim C",
              notes="Pre-registered rule, 20 paired crowds: the TabPFN planner beats no baseline (Holm, p = 1.0). At "
                    "constant velocity m0.4's safety it is 4.3 s slower (gap) and 3.0 s slower (collision rate). Even "
                    "the open-loop oracle could not beat every baseline by 2 s: the fastest baseline runs 1.5 s above "
                    "the 17.7 s time floor. Claim C is reported as a stress test.")
    strip(s, ["demo/seed3013_paths.png"], max_h=Inches(5.0))
    caption(s, "Showcase crowd 3013, chosen by rule (median split time): same crowd for every planner")

    s = slide(prs, "Why slower, and what comes next", kicker="Honest framing",
              notes="Slower, and not because of TabPFN's calibration: TabPFN with and without the scene model drove the "
                    "same; the caution setting sets the pace. At q 0.15 (20 confirmatory crowds) split 26.23 s, frozen "
                    "TabPFN 26.35 s (−0.12 s [−1.35, +1.25]); over 40 crowds +0.03 s [−1.08, +1.16], although split's boxes "
                    "are 0.26 m wider after the switch. The trade-off: LightGBM's narrower boxes buy about 1.2 s (40 crowds, "
                    "+1.24 s [+0.17, +2.34]), but after the switch they hold only 41% of people against split's 59%. "
                    "Faster because overconfident. All learned planners stay 3–5 s slower than constant velocity and "
                    "velocity obstacles at comparable safety: this box planner vetoes moves at one caution level. Next "
                    "work, not done: a planner that uses the range's shape, e.g. a chance-constrained cost.")
    bullets(s, ["TabPFN with and without the scene model drove the same",
                "The caution setting sets the pace, not calibration",
                "LightGBM ~1 s faster, but its boxes hold 41% of people, ours 59%",
                "Next: use the range's shape (chance-constrained cost), not done"])
    caption(s, "At a lower caution setting it matches constant velocity's speed with more clearance (exploratory, not significant)",
            top=Inches(6.4))
    caption(s, "In the densest crowd: about half the collisions of standard planners at similar speed (exploratory)")

    s = slide(prs, "When to use which", kicker="Honest framing",
              notes="Familiar place, many rows: LightGBM is as good, with narrower boxes. New place or new behaviour, few "
                    "rows: TabPFN keeps its promise better. Limits: two real scenes, a simulated crowd, no robot in the "
                    "real data, no statistical safety guarantee; calibrated ranges are for efficiency, a worst-case "
                    "floor is for safety.")
    bullets(s, ["Familiar place, many rows: LightGBM is as good", "New place or new behaviour, few rows: TabPFN is safer",
                "Two real scenes, simulated crowd, no safety guarantee claimed"])

    s = slide(prs, "On a robot", kicker="Deployment",
              notes="A small cached context in the loop, constant velocity as the safety floor, the cloud never in the "
                    "loop. Measured on a rented NVIDIA L4 GPU (docs/DEPLOYMENT.md): the TabPFN part of a replan takes "
                    "71 ms (split design, 48 query rows; 452 ms on the laptop CPU); the scene model is rebuilt in 318 ms "
                    "every 2 s, off the control path; the frozen fit is 490 ms once; the real loop is 225 ms per plan on "
                    "3 T5 seeds; GPU and CPU quantiles agree within 0.021 m. So 71 ms TabPFN and a 225 ms full loop fit the "
                    "0.4 s replan period; 10 Hz needs the background rebuild, which is designed but not built. No Jetson "
                    "was measured. Apple M3 Pro (MPS): live TabPFN 287–321 ms, frozen 130–135 ms per plan, MPS = CPU within "
                    "1.3 cm (Nooshin, tabpfn 10.2; docs/runs/mac_smoke.md). TabPFN-3.5 has 219 M parameters, 3.5-Fast 83.5 M. The 3.5 weights are "
                    "non-commercial: research prototype; a product needs the commercial licence or the Apache v2 weights.")
    _text(s, "TabPFN step 71 ms per replan on an NVIDIA L4 · full loop 225 ms · inside the 0.4 s replan period", Inches(0.6), Inches(6.35), Inches(12.1),
          Inches(0.5), 14, color=MUTED)
    caption(s, "One replan of stale prediction quadruples collision crowds (7.5% to 30%), so the GPU's 71 ms matters")
    image(s, "diagrams/12_deployment.png")

    OUT.parent.mkdir(exist_ok=True); prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
