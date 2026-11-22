"""Rebuild the working deck from media/ and results/. Run after every new figure or result.

    python tools/build_deck.py          -> deck/Foresight.pptx

Slides carry a title, one visual and at most a few words; explanations go in the speaker
notes. Missing figures become placeholder slides, so the deck always shows what is still to do.
Story (docs/STORY.md section 6): the robot's knowledge is a table of examples, so it adapts to new behaviour
and to a new body by swapping rows, with no training step. Three proofs, then where it breaks; backup after.
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
    """Main deck in the docs/STORY.md section 6 order, then backup. Numbers in notes and captions come from RESULTS.md."""
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H

    s = slide(prs, "Foresight", kicker="HackLab Berlin · Prior Labs track",
              notes="One prediction layer for a robot among people. TabPFN reads a table of what people just did and "
                    "returns, for each person, a range for where they will be in one and two seconds. The robot keeps "
                    "clear of the ranges. Nooshin Pourkamali, Darshak Panseriya.")
    _text(s, "Where will everyone be in two seconds, and how sure are we?", Inches(0.6), Inches(1.4), Inches(12), Inches(0.6), 22, color=MUTED)
    _text(s, "The robot's knowledge is a table of examples. It adapts to new behaviour, and to a new body, by swapping "
             "rows, with no training step.", Inches(0.6), Inches(1.95), Inches(12.1), Inches(0.7), 16, color=INK)
    image(s, "../docs/walkthrough/s6_all_boxes.png", top=Inches(2.7), max_h=Inches(4.0))
    caption(s, "Nooshin Pourkamali · Darshak Panseriya")

    s = slide(prs, "A robot needs the next two seconds, and how sure it is", kicker="Problem",
              notes="Walking straight is a strong baseline on real people (Schöller et al. 2020): in our claim A rows "
                    "(f4, +2 s, tuned constant-velocity filter; RESULTS.md, claim A table) it is off by 0.428 m "
                    "Hotel → ETH and 0.254 m ETH → Hotel. The hard part is knowing how wrong it might be, "
                    "and that is hardest in a place the robot has never seen, where an overconfident model is dangerous.")
    image(s, "diagrams/01_problem.png")

    s = slide(prs, "Scene → table → TabPFN → ranges → robot", kicker="Approach",
              notes="One row = one person, one moment, one look-ahead: their own motion, three nearest neighbours, the "
                    "robot, in the person's own frame. The label is how far they end up from walking straight. TabPFN "
                    "reads labelled rows as examples and returns quantiles in one pass, no training; its weights never change. "
                    "The robot's knowledge is the table: swap the rows and it knows a new place, behaviour or body.")
    _text(s, "The robot's knowledge is the table", Inches(0.6), Inches(1.4), Inches(12.1), Inches(0.5), 18, color=MUTED)
    image(s, "diagrams/02_pipeline.png", top=Inches(2.0), max_h=Inches(5.2))

    s = slide(prs, "The full loop", kicker="Demo · simulated crowd",
              notes="Same crowd for every robot: stop-and-go, reactive, velocity obstacles, ours (TabPFN split, the "
                    "ranges it planned around drawn) and the oracle that sees the true future. The crowd was chosen by a "
                    "fixed rule, not cherry-picked: crowd 3013 is one where every planner reached the goal and ours is "
                    "closest to its median time (25.8 s against a 26.40 s median; RESULTS.md). Five seconds in, every "
                    "person turns erratic, a behaviour none of the models has seen. Simulated; in PowerPoint the GIF "
                    "plays in slideshow mode.")
    image(s, "demo/seed3013.gif", max_h=Inches(5.15))
    caption(s, "Same crowd for every robot · chosen by a fixed rule (median time), not cherry-picked · turns erratic at 5 s")

    s = slide(prs, "Adapts in the loop when behaviour changes", kicker="Proof 1 · claim B",
              notes="20 fresh simulated crowds that turn erratic at 5 s; erratic was never in the bank. After the switch, "
                    "reading this scene's last completed rows (split) keeps the box holding the truth 59.3% of the time, "
                    "against 49.3% for frozen TabPFN (+10.0 points [+8.4, +11.6]) and 41.3% for LightGBM; interval "
                    "score 5.03 vs 6.34 (−1.31 [−1.55, −1.08]); error −0.057 m. Against LightGBM: 59.3% vs 41.3%, +18.0 points, of "
                    "which reading the scene alone adds ten points over the same TabPFN without it. Before the switch it costs a little. "
                    "No model reaches the 81% target. Against LightGBM it holds in three crowds (open plaza +6.6, crossing "
                    "+18.0, dense +19.7 points; GENERALISATION.md). Simulation; real data: a small gain (RESULTS.md). The picture: crowd 3013, TabPFN split against LightGBM on "
                    "all rows; each person's 5–95% box drawn 2 s earlier, dots where they actually were "
                    "(media/demo/calibration_seed3013.gif). This crowd ends at 60.1% vs 45.2% inside since the change. "
                    "Honest reading: TabPFN's boxes are wider (2.08 vs 1.48 m) and the interval score still favours it "
                    "(5.16 vs 6.53). TabPFN lags until about 10 s, when it has refitted on the scene, so the first column "
                    "shows LightGBM ahead (kept on purpose). Both are under the 81% target. "
                    "Fairness question (LightGBM never saw the scene): offline, a LightGBM refit on rows that include "
                    "the same scene still covered 12.3 points less at 80% [+7.7, +17.0], pinball −0.015 [−0.022, "
                    "−0.009], error level (+0.021 m [−0.009, +0.048]); results/live_variants.csv Setting B, t ≥ 20 s, "
                    "166 people, person bootstrap; post-hoc, exploratory. The clean adaptation number is TabPFN against "
                    "itself: +10.0 points from the scene rows.")
    image(s, "demo/calibration_seed3013_strip.png", max_h=Inches(5.15))
    caption(s, "This crowd: 60% vs 45% inside after the change; 20 crowds: 59% vs 41%")

    s = slide(prs, "Same core, another form factor", kicker="Proof 2 · highway cars",
              notes="The core, unchanged, on highway-env cars (4 lanes, 20 IDM/MOBIL cars; offline, simulated; "
                    "docs/HIGHWAY.md, rules fixed before any TabPFN run). H1 passed by its pre-set rule: TabPFN on 1,500 "
                    "rows matches LightGBM trained on 67 times more rows, with more honest ranges: 85.6% coverage, "
                    "interval score 3.57, against the pre-set LightGBM's 82.1% / 4.06 (−0.49 [−0.66, −0.34]). A "
                    "post-hoc check with well-fitted LightGBM settings shrinks that to a tie (−0.18 [−0.37, +0.00]; those "
                    "models cover only 69–72%). So say \"matches\", never \"beats\". Constant velocity plus band: "
                    "82.9% / 7.78. H2 passed: after traffic turns aggressive, a style "
                    "the bank never saw, split beats frozen (coverage 76.7% vs 62.5%, interval score −0.98 [−1.41, "
                    "−0.56]); about 80% of the gain is in the first 6 s after the switch. A pedestrian bank on cars gives "
                    "~50 m boxes, so each embodiment needs its own bank (D-35). No closed loop on cars.")
    image(s, "fig_highway.png", max_h=Inches(5.1))
    caption(s, "Simulated traffic, offline: 1,500 rows match LightGBM trained on 67x more, with more honest ranges · adapts after traffic turns aggressive")

    s = slide(prs, "Learns more per example", kicker="Proof 3 · five real scenes",
              notes="Leave one of five real scenes out (ETH, Hotel, Zara1, Zara2, Univ; docs/LOSO5.md, rule fixed before "
                    "the run). On the same 1,500 rows TabPFN beats LightGBM (interval score −0.130 [−0.154, −0.107], "
                    "clearly in 4 of 5 scenes). Against LightGBM trained on all rows of four scenes (about 15 times more) "
                    "it ties (−0.008 [−0.023, +0.007]); it is worse on Hotel and Zara1 and better on Univ. Say \"ties "
                    "with 15 times less data\". In the crowd simulator, TabPFN on 2,000 rows is about level with "
                    "LightGBM on 20,000 (results/context_size.csv). Zara1 and Zara2 share a camera. Hand-made interaction "
                    "features added nothing measurable; TabPFN reads them out of the raw rows (D-36, docs/FEATURES_ENG.md).")
    image(s, "fig_loso5.png", max_h=Inches(5.1))
    caption(s, "Five real scenes, each held out once: beats LightGBM on the same rows; ties one trained on 15x more")

    s = slide(prs, "TabPFN at every learned step", kicker="Result · chained TabPFN",
              notes="In the running loop TabPFN does three jobs: it predicts ranges (frozen bank model), adapts them "
                    "(recent-only scene model) and picks the move (a TabPFN classifier over the candidate moves, fitted "
                    "once on its teacher's choices). On 20 fresh crowds it matched its teacher, the box planner on the "
                    "same ranges: goal 100% vs 100%, time +0.55 s [−0.34, +1.65], gap +0.009 m, collision crowds +5 "
                    "points [−15, +25]. It picks the teacher's move at 87% of replans; the 0.4 s safety check and the "
                    "area check override it at 0.9%; 333 ms per plan on an NVIDIA L4. Honesty line: it matches its "
                    "teacher on point estimates; the collision interval is too wide at 20 crowds to rule out a "
                    "25-point worse rate. It imitates a planner that does not beat the baselines, so this shows TabPFN "
                    "can be the controller, not a better one. A fourth role, offline on real data: a TabPFN intent "
                    "classifier feeding per-manoeuvre range models (better-calibrated ranges for people who turn; it "
                    "does not forecast turns). The pictured crowd (3403) is the lowest test seed where both reached the "
                    "goal without contact, so it is not a typical crowd: the policy happened to keep more clearance there. What stays classical: the tracker (a Kalman filter), the "
                    "online offset and the hard checks (the 0.4 s constant-velocity check and staying in the area).")
    _text(s, "Predicts ranges (frozen bank) → adapts (scene model) → picks the move (classifier over candidates)\n"
             "Offline, real data: intent classifier → per-manoeuvre ranges", Inches(0.6), Inches(1.45), Inches(12.1),
          Inches(0.9), 16, color=INK)
    image(s, "demo/policy_seed3403_paths.png", top=Inches(2.35), max_h=Inches(4.35))
    caption(s, "Teacher (left) vs TabPFN policy (right), crowd 3403: lowest test seed where both reached the goal without "
               "contact · overall: matches its teacher on 20 crowds, safety checks override 0.9%, 333 ms per plan on an L4")

    s = slide(prs, "Slower than standard planners", kicker="Closed loop · claim C",
              notes="Pre-registered rule, 20 paired crowds: the TabPFN planner beats no baseline (Holm, p = 1.0). At "
                    "constant velocity m0.4's safety it is 4.3 s slower (gap) and 3.0 s slower (collision rate). Even "
                    "the open-loop oracle could not beat every baseline by 2 s: the fastest baseline runs 1.5 s above "
                    "the 17.7 s time floor. Claim C is reported as a stress test.")
    image(s, "fig_t5.png", max_h=Inches(5.15))
    caption(s, "Left: speed against safety per planner over its dial (20 paired crowds) · right: in-loop calibration")

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
              notes="Few examples, or behaviour that just changed: TabPFN (in-loop crowds +18 points against LightGBM "
                    "after the change; cars +14 points from the scene). Plenty of familiar data: a well-fitted LightGBM is "
                    "as good (five real scenes: a tie against LightGBM on 15 times more rows; cars, simulated and offline: "
                    "a tie against well-fitted settings). The next lever is the planner: better ranges did not change how "
                    "the box planner drives. Limits: crowds and cars are simulated, cars offline, no robot in the real "
                    "data, no statistical safety guarantee; calibrated ranges are for efficiency, a worst-case floor is "
                    "for safety.")
    bullets(s, ["Few examples, or behaviour that just changed: TabPFN",
                "Plenty of familiar data: a well-fitted LightGBM is as good",
                "The next lever is the planner"])

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

    s = slide(prs, "Backup", kicker="Backup",
              notes="Backup slides: how one decision is made, how we tested, the two-scene claim A result, and the "
                    "intent chain on real people.")

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

    s = slide(prs, "Honest ranges in a new place, on two scenes", kicker="Result · claim A (two scenes, not five)",
              notes="Real Zurich pedestrians, context from one scene, test on the other, 5 context samples, paired person "
                    "bootstrap. Hotel → ETH: TabPFN's 5-95% box holds the truth 74.8% (LightGBM on all rows 57.7%); "
                    "interval score −0.303 [−0.425, −0.184] against LightGBM. ETH → Hotel: 86.4% vs 74.9%, interval "
                    "score level with LightGBM on all rows. Direction-dependent; in the hard direction TabPFN still "
                    "under-covers (75% against the 81% target). Point error level with LightGBM, slightly better than tuned CV. On five scenes (leave one out, "
                    "docs/LOSO5.md): edge strongest when data is scarce; ties a rival trained on 15x more varied data "
                    "(pooled −0.008 [−0.023, +0.007]); on the same 1,500 rows it beats LightGBM (−0.130). Say it as: better "
                    "calibrated in a new place on two scenes, not on five; ties with 15 times less data.")
    image(s, "fig_f4_claim_a.png")

    s = slide(prs, "TabPFN at two chained layers", kicker="Result · intent",
              notes="A TabPFN classifier gives manoeuvre probabilities (straight, stop, turn left or right); per-manoeuvre "
                    "TabPFN range models give ranges, mixed by those probabilities. Real Hotel → ETH: on people who turn, "
                    "the interval score improves by −0.263 [−0.481, −0.075] (73 rows, 37 people) and coverage rises from "
                    "45.7% to 62.6%; overall level. It does NOT forecast turns: turn recall is 0. The gain is calibrated "
                    "widening from the mixture. Say 'better-calibrated ranges for people who turn'. In the simulator "
                    "intent is random by construction and the chain is not adopted (D-34). Offline only.")
    image(s, "fig_intent_layer.png")

    OUT.parent.mkdir(exist_ok=True); prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
