"""Finalist deck from docs/DECK_OUTLINE.md: ten talk slides, a closing slide, a divider and the backup slides,
in the 3D video's look (warm page, white cards, one orange accent). The spoken text, the time and
the speaker are in the speaker notes. Numbers are the ones in docs/STORY.md.

    python tools/build_outline_deck.py            # deck/Foresight_finalist.pptx (clips play in the slide show)
    python tools/build_outline_deck.py --static   # deck/Foresight_finalist_static.pptx: each clip replaced by
                                                  # its last frame, for a PDF export or a machine that cannot play them

Pictures made for the deck are in media/deck/ (frames of the 2:46 3D video, the two panels of
fig_t5.png, the last frame of each clip, the 33 s street clip).
"""
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
W, H, M = 13.333, 7.5, 0.6
CW = W - 2 * M                                    # content width


def rgb(h):
    return RGBColor.from_string(h)


BG, CARD, INK, MUTED, RULE = rgb("F5F1EA"), rgb("FFFFFF"), rgb("1C2226"), rgb("666C72"), rgb("E2DBD0")
ACCENT, BLUE, GREEN, GREY = rgb("C24E1C"), rgb("2F6FDB"), rgb("1F9D6B"), rgb("A3A8AD")
OK_BG, OK_INK, NO_BG, NO_INK, SOFT = rgb("DCEFE3"), rgb("17603F"), rgb("F7E1D8"), rgb("8A3511"), rgb("FBEFE8")
FONT = "Arial"
DK = "media/deck/"
G = "media/demo/"
STATIC = "--static" in sys.argv
REPO = "github.com/shiinalight/Robodetect"


# ---------------------------------------------------------------- drawing helpers
def text(slide, x, y, w, h, s, size, bold=False, color=INK, align=None, anchor=None, spc=None, gap=0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if anchor:
        tf.vertical_anchor = anchor
    for i, part in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = align
        if gap:
            p.space_after = Pt(gap)
        r = p.add_run()
        r.text = part
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
        r.font.color.rgb = color
        if spc:
            r._r.get_or_add_rPr().set("spc", str(spc))
    return tb


def box(slide, x, y, w, h, fill=CARD, line=RULE, radius=0.05, kind=MSO_SHAPE.ROUNDED_RECTANGLE, lw=0.75):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE and radius is not None:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    return s


def label_box(slide, x, y, w, h, title, sub=None, fill=CARD, line=RULE, ink=INK, sub_ink=MUTED, size=17, sub_size=12):
    box(slide, x, y, w, h, fill=fill, line=line, radius=0.08)
    if sub:
        text(slide, x + 0.15, y + 0.14, w - 0.3, 0.4, title, size, bold=True, color=ink, align=PP_ALIGN.CENTER)
        text(slide, x + 0.15, y + 0.14 + size / 72 * 1.45, w - 0.3, h - 0.6, sub, sub_size, color=sub_ink, align=PP_ALIGN.CENTER)
    else:
        text(slide, x + 0.15, y, w - 0.3, h, title, size, bold=True, color=ink, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def pic(slide, rel, x, y, w, h, cover=False):
    """Place a picture inside the box (fit, centred) or filling it (cover, cropped evenly)."""
    path = ROOT / rel
    iw, ih = Image.open(path).size
    if cover:
        p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
        if iw / ih > w / h:
            c = (1 - (w / h) / (iw / ih)) / 2
            p.crop_left = p.crop_right = c
        else:
            c = (1 - (iw / ih) / (w / h)) / 2
            p.crop_top = p.crop_bottom = c
        return p
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def clip(slide, name, x, y, w, h, pad=0.12):
    """A clip from media/demo on a white card; in the static deck, its last frame."""
    box(slide, x, y, w, h)
    rel = DK + name + "_last.png" if STATIC else G + name + ".gif"
    pic(slide, rel, x + pad, y + pad, w - 2 * pad, h - 2 * pad)


def bars(slide, x, y, w, rows, vmax, label_w=1.75, pitch=0.66, bar_h=0.3):
    """Horizontal bars. rows: (label, small label, value, value text, colour)."""
    bw = w - label_w - 0.85
    for i, (lab, sub, v, vt, col) in enumerate(rows):
        yy = y + i * pitch
        text(slide, x, yy - 0.05, label_w - 0.12, 0.26, lab, 12, bold=True, align=PP_ALIGN.RIGHT)
        if sub:
            text(slide, x, yy + 0.17, label_w - 0.12, 0.22, sub, 9.5, color=MUTED, align=PP_ALIGN.RIGHT)
        box(slide, x + label_w, yy, max(0.05, bw * v / vmax), bar_h, fill=col, line=None, radius=0.04)
        text(slide, x + label_w + bw * v / vmax + 0.1, yy, 0.8, bar_h, vt, 14, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    return y + len(rows) * pitch


def pill(slide, x, y, w, s, good=True, h=0.46, size=14):
    box(slide, x, y, w, h, fill=OK_BG if good else NO_BG, line=None, radius=0.1)
    text(slide, x + 0.16, y, w - 0.32, h, ("✓  " if good else "✗  ") + s, size, bold=True, color=OK_INK if good else NO_INK,
         anchor=MSO_ANCHOR.MIDDLE)


def chevron(slide, x, y, size=0.3, down=False):
    s = box(slide, x, y, size, size, fill=GREY, line=None, kind=MSO_SHAPE.CHEVRON)
    if down:
        s.rotation = 90
    return s


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.n = 0

    def slide(self, kicker=None, title=None, notes="", meta=None, footer=True, title_size=28):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = BG
        s.notes_slide.notes_text_frame.text = (meta + "\n\n" if meta else "") + notes
        if kicker:
            text(s, M, 0.4, CW, 0.28, kicker, 13, bold=True, color=ACCENT)
        if title:
            text(s, M, 0.7, CW, 1.0, title, title_size, bold=True)
        if footer:
            text(s, M, 7.05, 6, 0.25, "Foresight  ·  Prior Labs track", 10, color=MUTED)
            text(s, W - M - 1, 7.05, 1, 0.25, str(self.n), 10, color=MUTED, align=PP_ALIGN.RIGHT)
        return s


# ---------------------------------------------------------------- the talk
def talk(d):
    # 1 title
    s = d.slide(footer=False, meta="Slide 1 · Title · 15 s · Darshak",
                notes="Say: A delivery robot shares the street with people, cars and bikes. Foresight tells it where they'll be next, and how much to trust that.")
    pic(s, "media/stills/3d_range_every_agent.jpg", 0, 0, W, H, cover=True)
    box(s, 0, 4.46, W, H - 4.46, fill=BG, line=None, kind=MSO_SHAPE.RECTANGLE)
    text(s, M, 5.1, CW, 0.95, "Foresight", 56, bold=True)
    text(s, M, 6.05, CW, 0.45, "Where will everyone be in two seconds, and how sure are we?", 24)
    text(s, M, 6.72, 8, 0.3, "Nooshin Pourkamali  ·  Darshak Panseriya", 15, color=MUTED)
    text(s, W - M - 4, 6.72, 4, 0.3, "Prior Labs track", 15, bold=True, color=ACCENT, align=PP_ALIGN.RIGHT)

    # 2 problem
    s = d.slide("The problem", "Robots guess with a straight line and a fixed margin", meta="Slide 2 · The problem · 30 s · Darshak",
                notes="Say: today's options are to assume everyone walks straight, add one margin, or stop when someone is close. Stock ROS 2 Nav2 plans around where people are now. Learned forecasters do better, but once trained they are frozen. Too narrow a margin and the robot hits someone; too wide and it freezes.")
    pic(s, DK + "3d_problem.jpg", M, 1.75, 7.55, 4.25, cover=True)
    x = 8.5
    text(s, x, 1.7, 4.3, 1.1, "0.93 m", 66, bold=True, color=ACCENT)
    text(s, x, 2.85, 4.2, 0.9, "how far \"keep walking straight\" is off after 2 s in our simulated crowd", 15)
    text(s, x, 3.62, 4.2, 0.4, "0.25–0.43 m on real Zurich pedestrians", 13, color=MUTED)
    for i, (a, b) in enumerate([("Margin too narrow", "the robot hits someone"), ("Margin too wide", "the robot freezes")]):
        yy = 4.25 + i * 0.92
        box(s, x, yy, 4.23, 0.78, radius=0.08)
        text(s, x + 0.2, yy + 0.11, 3.9, 0.3, a, 14, bold=True)
        text(s, x + 0.2, yy + 0.41, 3.9, 0.3, b, 13, color=MUTED)
    text(s, M, 6.3, CW, 0.4, "Learned forecasters do better, but once trained they are frozen.", 18, bold=True)

    # 3 idea
    s = d.slide("The idea", "The robot's knowledge is a table", meta="Slide 3 · The idea · 30 s · Darshak",
                notes="Say: each person, at each moment, is one row: their motion, neighbours, the robot, obstacles. TabPFN reads a few thousand example rows and answers in one pass, with no training. To adapt to a new place, a new behaviour or a new body, change the rows.\nDetail: frozen bank of 2,000 rows plus up to 250 rows of the current scene, refit every 2 s; 5–95% range per person at +1 and +2 s; weights never change on the robot.")
    tw, ty = 5.9, 1.75                                             # numbers: the step-2 card of the 3D video
    box(s, M, ty, tw, 3.05)
    cols = [("agent", 1.62), ("speed\nm/s", 0.7), ("turning\n°/s", 0.74), ("nearest\nm", 0.74), ("robot\nm", 0.64), ("within\n3 m", 0.68),
            ("where\nin 2 s?", 0.78)]
    rows = [("pedestrian 14", "1.18", "−17.7", "1.6", "13.9", "1", "?"), ("pedestrian 2", "1.43", "13.5", "1.8", "16.1", "1", "?"),
            ("pedestrian 4", "1.84", "2.5", "11.6", "27.6", "0", "?"), ("pedestrian 9", "1.36", "−56.6", "6", "6.8", "0", "?")]
    box(s, M + 0.08, ty + 0.78, tw - 0.16, 0.52, fill=SOFT, line=None, radius=0.04)
    xx = M
    for j, (head, cwid) in enumerate(cols):
        al = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
        text(s, xx + 0.14, ty + 0.16, cwid - 0.22, 0.5, head, 10.5, color=MUTED, align=al)
        for i, r in enumerate(rows):
            text(s, xx + 0.14, ty + 0.78 + i * 0.54, cwid - 0.22, 0.52, r[j], 12.5, bold=(i == 0 or j == 0), color=ACCENT if j == 6 else INK,
                 align=al, anchor=MSO_ANCHOR.MIDDLE)
        xx += cwid
    for i in range(2, 4):
        box(s, M + 0.15, ty + 0.78 + i * 0.54 - 0.01, tw - 0.3, 0.012, fill=RULE, line=None, kind=MSO_SHAPE.RECTANGLE)
    text(s, M, ty + 3.2, tw, 0.3, "One row per agent, every 0.4 s: its motion, its neighbours, the robot, obstacles", 12, color=MUTED)
    text(s, M, ty + 3.65, tw, 0.4, "Example rows are the same table with the last column known.", 14, bold=True)
    x, w = 7.0, W - M - 7.0
    half = (w - 0.25) / 2
    label_box(s, x, 1.75, half, 1.12, "Example bank", "2,000 example rows · frozen")
    label_box(s, x + half + 0.25, 1.75, half, 1.12, "This scene", "up to 250 newest rows · swapped every 2 s", fill=SOFT, line=ACCENT)
    chevron(s, x + w / 2 - 0.15, 3.0, down=True)
    label_box(s, x, 3.43, w, 1.05, "TabPFN", "reads the rows and answers in one pass · weights never change", fill=ACCENT, line=None,
              ink=CARD, sub_ink=rgb("FBE3D8"))
    chevron(s, x + w / 2 - 0.15, 4.61, down=True)
    label_box(s, x, 5.04, w, 0.81, "A range for every agent", "5–95%, 1 s and 2 s ahead", size=15, sub_size=11.5)
    text(s, M, 6.3, CW, 0.45, "No training step. Adapting = swapping rows.", 22, bold=True, color=ACCENT)

    # 4 four steps
    s = d.slide("How it reads one person", "From sensor dots to a range, in four steps", meta="Slide 4 · How it reads one person · 15 s · Nooshin",
                notes="Say: a filter smooths the noisy track; the moment becomes a row; TabPFN predicts the correction to walking straight as a 5–95% range; the range is sized so the truth lands inside as often as promised.\nKeep it to 15 s: the video already showed these steps.")
    steps = [("3d_step1_watch.jpg", "Watch", "A filter turns noisy sensor dots into a smooth track."),
             ("3d_step2_row.jpg", "Row", "Every 0.4 s, each agent's situation becomes one row of numbers."),
             ("3d_step3_tabpfn.jpg", "TabPFN", "Reads example rows and predicts the correction to walking straight, as a 5–95% range."),
             ("3d_step4_size.jpg", "Size", "The range is stretched or shrunk until the truth lands inside as often as promised.")]
    pw, tw = 3.5, 2.15                                             # picture, then number, word and sentence to its right
    ph = pw / 1.44
    cell = (CW - 0.3) / 2
    for i, (f, t, body) in enumerate(steps):
        x, y = M + (i % 2) * (cell + 0.3), 1.72 + (i // 2) * (ph + 0.22)
        pic(s, DK + f, x, y, pw, ph, cover=True)
        box(s, x + pw + 0.25, y + 0.1, 0.46, 0.46, fill=ACCENT, line=None, kind=MSO_SHAPE.OVAL)
        text(s, x + pw + 0.25, y + 0.1, 0.46, 0.46, str(i + 1), 16, bold=True, color=CARD, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + pw + 0.85, y + 0.1, tw - 0.85, 0.46, t, 20, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + pw + 0.25, y + 0.78, tw - 0.15, 1.6, body, 13.5, color=MUTED)

    # 5 street
    s = d.slide("Watch it on a street", "A street run the model has never seen", meta="Slide 5 · Watch it on a street · 35 s · Nooshin",
                notes="Say: someone stops dead, a car brakes for the crossing, someone crosses away from the crossing, rush hour changes the crowd. Every agent gets a range.\nThe clip is 33 s, silent; click it to play. It is the street part of the 3D video (1:05 to 1:38).\nThe street's ranges come from a fixed example set: don't say it adapts live here. Slide 6 shows adaptation.")
    vx, vy, vw = M, 1.75, 8.35
    vh = vw * 9 / 16
    if STATIC:
        pic(s, DK + "3d_street_stop.jpg", vx, vy, vw, vh, cover=True)
    else:
        s.shapes.add_movie(str(ROOT / DK / "street_clip.mp4"), Inches(vx), Inches(vy), Inches(vw), Inches(vh),
                           poster_frame_image=str(ROOT / DK / "3d_street_stop.jpg"), mime_type="video/mp4")
    text(s, vx, vy + vh + 0.1, vw, 0.3, "Simulated street  ·  robot route scripted  ·  ranges from TabPFN, fixed set of example rows", 11, color=MUTED)
    x = vx + vw + 0.4
    w = W - M - x
    events = [("A person stops suddenly", "the range has to stay with them"), ("A car brakes hard at the crossing", "where it will be in 2 s changes fast"),
              ("Someone crosses mid-street", "no usual path predicts this"), ("Rush hour", "more people, more cars, a new pace")]
    for i, (a, b) in enumerate(events):
        yy = 1.78 + i * 1.02
        text(s, x, yy, w, 0.3, a, 13.5, bold=True)
        text(s, x, yy + 0.4, w, 0.3, b, 12, color=MUTED)
        box(s, x, yy + 0.84, w, 0.012, fill=RULE, line=None, kind=MSO_SHAPE.RECTANGLE)
    text(s, x, 6.0, w, 0.35, "Every agent gets a range.", 16, bold=True)

    # 6 proof 1
    s = d.slide("Proof 1 · Adapts to dynamic environments", "Reading the scene kept more people inside their ranges",
                meta="Slide 6 · Proof 1 · 35 s · Darshak",
                notes="Say: in our robot loop the crowd turns erratic five seconds in. The TabPFN that reads the last seconds kept 59% of people inside their 2 s range; LightGBM kept 41%. Ten of those points come from reading the scene alone. Ahead in all three crowds (+6.6, +18.0, +19.7 pts).\nThe clip is one crowd (seed 3013): its counters end near 60% and 45%; the bars are all 20 crowds. Target is 81%: closer, not there. If asked whether LightGBM had the scene rows: backup B1.")
    y0, ch = 1.75, 4.7
    clip(s, "calibration_seed3013", M, y0, 7.5, ch)
    text(s, M, y0 + ch + 0.08, 7.5, 0.3, "One crowd of the 20: about 60% vs 45% inside  ·  green dot = inside its range, red = outside", 11, color=MUTED)
    x, w = 8.4, W - M - 8.4
    box(s, x, y0, w, ch)
    text(s, x + 0.25, y0 + 0.2, w - 0.5, 0.5, "People inside their 2 s range after the crowd changed", 13, bold=True)
    yb = bars(s, x + 0.2, y0 + 0.95, w - 0.4, [("TabPFN", "bank + this scene", 59.3, "59%", ACCENT), ("TabPFN", "bank only", 49.3, "49%", rgb("E7A37F")),
                                               ("LightGBM", "all rows", 41.3, "41%", BLUE)], 100, label_w=1.45)
    pill(s, x + 0.25, yb + 0.02, w - 0.5, "+18 points, no retraining")
    text(s, x + 0.25, yb + 0.62, w - 0.5, 0.6, "+10 points from reading the scene alone\nAhead in 3 of 3 crowds", 12.5, gap=3)
    text(s, x + 0.25, y0 + ch - 0.36, w - 0.5, 0.25, "Simulated robot loop · 20 paired crowds · target 81%", 9.5, color=MUTED)

    # 7 proof 2
    s = d.slide("Proof 2 · Works across modalities", "The same core predicts highway traffic", meta="Slide 7 · Proof 2 · 30 s · Darshak",
                notes="Say: same features and models, a small adapter for cars. 1,500 example rows match LightGBM trained on 67 times more, and our ranges held the truth 86% of the time against 69–82%. When drivers turn aggressive, reading the scene adds 14 points.\nThe clip is one road of the five (77 against 67 of 100 forecasts held); all five roads: 77% against 63%.\nSimulated traffic (highway-env), offline. Say 'matches', never 'beats'. Each body needs its own example bank.")
    y0, ch = 1.75, 4.5
    clip(s, "highway", M, y0, 7.5, ch)
    text(s, M, y0 + ch + 0.08, 7.5, 0.3, "This road: 77 vs 67 of 100 forecasts held  ·  all five roads: 77% vs 63%", 11, color=MUTED)
    box(s, x, y0, w, ch)
    text(s, x + 0.25, y0 + 0.2, w - 0.5, 0.3, "Cars instead of people", 13, bold=True)
    text(s, x + 0.25, y0 + 0.52, w - 0.5, 0.5, "Interval score: range width plus a penalty for misses. Lower is better.", 10.5, color=MUTED)
    yb = bars(s, x + 0.2, y0 + 1.3, w - 0.4, [("TabPFN", "1,500 rows", 3.57, "3.57", ACCENT), ("LightGBM", "100,440 rows", 3.74, "3.74", BLUE)],
              4.4, label_w=1.45)
    pill(s, x + 0.25, yb + 0.02, w - 0.5, "Same result, 67× less data")
    text(s, x + 0.25, yb + 0.62, w - 0.5, 0.9, "Truth inside the range: 86% vs 69–82%\n+14 points when drivers turn aggressive", 12.5, gap=3)
    text(s, x + 0.25, y0 + ch - 0.36, w - 0.5, 0.25, "Simulated highway (highway-env) · offline", 9.5, color=MUTED)

    # 8 proof 3
    s = d.slide("Proof 3 · Works with very little data", "On five real scenes, TabPFN gets more from each example", meta="Slide 8 · Proof 3 · 30 s · Darshak",
                notes="Say: ETH, Hotel, Zara 1 and 2, Univ: train on four, test on the fifth. TabPFN beats the fixed-margin guess from slide 2 in all five scenes. On the same rows it beats LightGBM in four scenes and ties on Hotel. LightGBM needs about 15 times the data to catch up.\nThe clip is one 16 s window per scene, an illustration; the numbers below are the test (interval score, mean of the five scenes).\nAgainst constant velocity plus a fixed band: −0.365 [−0.405, −0.328] pooled (forecast quality only). Interval score −0.130 pooled at equal rows; −0.008 (tie) at 15x. Zara1 and Zara2 share one camera. Scene by scene: backup B10.")
    gh = 3.3
    clip(s, "scenes5", M, 1.6, CW, gh, pad=0.1)
    text(s, M, 1.6 + gh + 0.06, CW, 0.25, "One 16 s window per scene  ·  1,500 example rows from the other four scenes  ·  small print below: interval score, lower is better", 10.5, color=MUTED)
    stats = [("5 of 5 scenes", "better than the fixed-margin guess", "1.36 vs 1.72"),
             ("4 of 5 scenes", "ahead of LightGBM on the same rows", "level on Hotel  ·  1.36 vs 1.49"),
             ("15× less data", "same result as LightGBM", "1,500 rows vs about 22,000  ·  1.36 vs 1.37")]
    gap = 0.2
    cw = (CW - 2 * gap) / 3
    for i, (big, body, small) in enumerate(stats):
        xx, yy = M + i * (cw + gap), 5.32
        box(s, xx, yy, cw, 1.5, radius=0.08)
        text(s, xx + 0.22, yy + 0.12, cw - 0.44, 0.5, big, 26, bold=True, color=ACCENT)
        text(s, xx + 0.22, yy + 0.68, cw - 0.44, 0.3, body, 13, bold=True)
        text(s, xx + 0.22, yy + 1.08, cw - 0.44, 0.3, small, 11, color=MUTED)

    # 9 where it stops
    s = d.slide("Where it stops", "Better forecasts, a simple planner", meta="Slide 9 · Where it stops · 30 s · Darshak",
                notes="Say: the forecasts are better, but our planner is simple on purpose: it only asks yes or no. At the same safety it arrives three to four seconds after standard planners. Next, it weighs how likely each spot is. The robot loop and the cars are simulated; the five pedestrian scenes are real.\nThe five-scene tie is proof 3 (data efficiency), and per-body banks are slide 10: neither is a limit.\nIf asked how the forecast beats a fixed margin while the constant-velocity planner beat the robot: the ranges are better in every real scene, but this planner turns each range into a yes/no box, so its caution setting sets the pace. Backup B7 for 'why slower' (every planner, time against closest gap).")
    y0, ch = 1.75, 4.3
    cw = (CW - 0.25) / 2
    box(s, M, y0, cw, ch)
    text(s, M + 0.3, y0 + 0.38, cw - 0.6, 0.4, "The limit: a simple planner", 22, bold=True)
    text(s, M + 0.3, y0 + 1.1, cw - 0.6, 0.6, "It only asks yes or no: is this move inside a range? The shape of the range is thrown away.", 13.5)
    text(s, M + 0.3, y0 + 1.95, cw - 0.6, 0.3, "Seconds behind the \"keep going straight\" planner, at the same safety", 11, color=MUTED)
    yb = bars(s, M + 0.2, y0 + 2.4, cw - 0.5, [("Same closest gap", None, 4.3, "+4.3 s", ACCENT), ("Same collision rate", None, 3.0, "+3.0 s", ACCENT)], 5.2,
              label_w=1.95, pitch=0.5)
    pill(s, M + 0.3, y0 + ch - 0.75, cw - 0.6, "Same safety, 3–4 s slower", good=False)
    x2 = M + cw + 0.25
    box(s, x2, y0, cw, ch)
    text(s, x2 + 0.3, y0 + 0.38, cw - 0.6, 0.4, "Next: use the whole range", 22, bold=True)
    text(s, x2 + 0.3, y0 + 1.1, cw - 0.6, 0.6, "A planner that weighs how likely each spot is, not just yes or no.", 13.5)
    ix, iy = x2 + 0.55, y0 + 2.0                                   # today: one flat box
    box(s, ix, iy, 1.9, 1.05, fill=SOFT, line=ACCENT, radius=0.03, lw=1.5)
    text(s, ix, iy + 1.12, 1.9, 0.25, "today: inside or outside", 11, color=MUTED, align=PP_ALIGN.CENTER)
    chevron(s, ix + 2.2, iy + 0.38)
    jx = ix + 2.8                                                  # next: likelihood falls off from the centre
    for k, (pad, col) in enumerate([(0.0, "FBEFE8"), (0.17, "F3C9B2"), (0.33, "E58F62"), (0.46, "C24E1C")]):
        box(s, jx + pad * 1.6, iy + pad * 0.95, 1.9 - 2 * pad * 1.6, 1.05 - 2 * pad * 0.95, fill=rgb(col), line=None, kind=MSO_SHAPE.OVAL)
    text(s, jx, iy + 1.12, 1.9, 0.25, "next: how likely each spot is", 11, color=MUTED, align=PP_ALIGN.CENTER)
    box(s, x2 + 0.3, y0 + ch - 0.75, cw - 0.6, 0.46, fill=rgb("EFEBE4"), line=None, radius=0.1)
    text(s, x2 + 0.46, y0 + ch - 0.75, cw - 0.9, 0.46, "A plan, not a result: nothing measured yet", 14, bold=True, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    text(s, M, 6.3, CW, 0.3, "Scope: robot loop and cars simulated  ·  five real pedestrian scenes  ·  TabPFN-3.5 weights are non-commercial", 12, color=MUTED)

    # 10 what it's for
    s = d.slide("What it's for", "A forecasting layer you re-target by swapping rows", meta="Slide 10 · What it's for · 35 s · Darshak",
                notes="Say: Foresight sits under whatever planner a team already runs. New site, new crowd behaviour, new vehicle: add rows, no training run. One example bank per body. Few examples or a sudden change: use TabPFN. Plenty of familiar data: a well-fitted LightGBM does as well. 71 ms per replan on an NVIDIA L4. Next: a planner that uses the full range, a closed loop for cars, real robot logs.\nClose: every number in this talk comes from a script in the repo.\nThe safety check is a 0.4 s look-ahead that assumes people keep walking straight; it is a check, not a guarantee. 'New body: new bank' is the offer, not a result: we tested pedestrians (real and simulated) and simulated cars.")
    bw, bg_ = 2.62, 0.55
    names = [("Sensors", "tracks of people and vehicles"), ("Foresight", "a range for every agent"), ("Your planner", "whatever you already run"),
             ("Safety check", "rules out moves too close in the next 0.4 s")]
    for i, (a, b) in enumerate(names):
        xx = M + i * (bw + bg_)
        hot = i == 1
        label_box(s, xx, 1.8, bw, 1.15, a, b, fill=ACCENT if hot else CARD, line=None if hot else RULE, ink=CARD if hot else INK,
                  sub_ink=rgb("FBE3D8") if hot else MUTED)
        if i < 3:
            chevron(s, xx + bw + (bg_ - 0.3) / 2, 2.22)
    fx = M + bw + bg_                                              # the bank row hangs under the Foresight box
    chevron(s, fx + bw / 2 - 0.15, 3.07, down=True).rotation = 270
    box(s, M, 3.5, CW, 1.2, fill=SOFT, line=ACCENT, radius=0.08)
    text(s, M + 0.3, 3.5, 3.6, 1.2, "Example bank\nSwap it, don't retrain", 15, bold=True, anchor=MSO_ANCHOR.MIDDLE, gap=2)
    chips = ["Pedestrians", "Cars", "New body: new bank"]
    cx = M + 4.0
    cwid = (CW - 4.0 - 0.3 - 2 * 0.2) / 3
    for i, c in enumerate(chips):
        box(s, cx + i * (cwid + 0.2), 3.78, cwid, 0.64, radius=0.1, line=ACCENT if i < 2 else GREY)
        text(s, cx + i * (cwid + 0.2), 3.78, cwid, 0.64, c, 14, bold=True, color=INK if i < 2 else MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    gw = (CW - 0.25) / 2
    for i, (a, b) in enumerate([("Few example rows, or behaviour just changed", "Use TabPFN"), ("Plenty of familiar data", "A well-fitted LightGBM is as good")]):
        xx = M + i * (gw + 0.25)
        box(s, xx, 4.95, gw, 1.0, radius=0.08)
        text(s, xx + 0.25, 5.07, gw - 0.5, 0.3, a, 12.5, color=MUTED)
        text(s, xx + 0.25, 5.4, gw - 0.5, 0.4, b, 19, bold=True, color=ACCENT if i == 0 else INK)
    text(s, M, 6.25, CW, 0.3, "71 ms per replan on an NVIDIA L4  ·  one example bank per body  ·  every number in this talk comes from a script in the repo",
         12.5, color=MUTED)

    # 11 close
    s = d.slide(footer=False, meta="Closing slide · stays up during questions", notes="Leave this up for questions. Backup slides follow.")
    x, w = 7.35, W - M - 7.35
    lines = [("+18 points", "Adapts when behaviour changes, no retraining", ACCENT), ("67× less data", "Cars: same result as LightGBM", ACCENT),
             ("15× less data", "Five real scenes: same result as LightGBM", ACCENT),
             ("3–4 s slower", "Where it stops: a yes/no planner, at the same safety", INK)]
    for i, (big, body, col) in enumerate(lines):
        yy = 1.45 + i * 1.22
        text(s, x, yy, w, 0.5, big, 26, bold=True, color=col)
        text(s, x, yy + 0.55, w, 0.3, body, 13.5, color=MUTED)
        if i < 3:
            box(s, x, yy + 1.02, w, 0.012, fill=RULE, line=None, kind=MSO_SHAPE.RECTANGLE)
    text(s, M, 1.7, 6, 0.95, "Foresight", 52, bold=True)
    text(s, M, 2.85, 5.9, 1.6, "The robot's knowledge is a table of examples. It adapts to new behaviour, and to a new body, by swapping rows, with no training step.", 19)
    text(s, M, 4.7, 5.9, 0.4, REPO, 18, bold=True, color=ACCENT)
    text(s, M, 5.15, 5.9, 0.3, "Every number comes from a script in the repo.", 13, color=MUTED)
    text(s, M, 6.3, 5.9, 0.6, "Nooshin Pourkamali  ·  Darshak Panseriya\nThank you, Prior Labs", 14, gap=3)


# ---------------------------------------------------------------- backup slides
BACKUPS = [
    dict(q="LightGBM never saw the scene. Is proof 1 fair?", pics=["media/fig_live_variants_cp.png"],
         a="Scene rows add 10 points; a LightGBM refit on rows from the same scene still covers about 12 points less (offline)",
         notes="Offline, simulated, after the switch: +12.3 pts 80% coverage [+7.7, +17.0], pinball −0.015, error level (results/live_variants.csv, post-hoc)."),
    dict(q="Did you tune until it worked?",
         bullets=["Every pass rule was committed before its run (pre-registration with amendments)", "Paired crowds, bootstrap intervals, Holm correction",
                  "Claim C failed and claim A narrowed under those rules", "Failures stay in the README"],
         notes="Commit hashes for each rule are in RESULTS.md and docs/DECISIONS.md."),
    dict(q="What data?", pics=["media/fig_f4_data_cards.png"],
         a="Five real scenes (ETH, Hotel, Zara 1–2, Univ) · three simulated crowds · highway-env traffic",
         notes="Real data: Pellegrini 2009 (ETH, Hotel), Lerner 2007 (UCY), read through OpenTraj. Annotated every 0.4 s, the step we replan on."),
    dict(q="How much data does TabPFN need?", pics=["media/fig_f4_context_size.png"], a="2,000 rows tie LightGBM on 20,000",
         notes="Crossing crowd, +2 s, f4 features, 3 seeds (results/context_size.csv)."),
    dict(q="Ranges in a new place?", pics=["media/fig_f4_claim_a.png"],
         a="Better than LightGBM on two scenes; a tie on five when LightGBM gets all the rows",
         notes="Two scenes: 75% vs 58% box coverage (Hotel to ETH). Five scenes: −0.008, a tie."),
    dict(q="Can it predict turns?", pics=["media/fig_intent_layer.png"],
         a="Wider ranges for people who turn (46% → 63% inside, real data); it does not call turns",
         notes="Intent chain: TabPFN classifier, then per-manoeuvre ranges. Offline only; turn recall is 0."),
    dict(q="Why was the robot slower?", pics=[DK + "fig_t5_left.png", "media/fig_f4_oracle_gate.png"],
         a="At the same caution setting every learned forecaster drives the same; even true future positions gain under 2 s",
         notes="Left: every planner, time to goal against closest gap, 20 paired crowds (T5). Right: the oracle gate. Split − frozen TabPFN −0.12 s [−1.35, +1.25] at q 0.15. The box veto throws away the range's shape."),
    dict(q="Fast enough? Feature engineering? Why these choices?",
         bullets=["Speed: 71 ms per replan on an NVIDIA L4, 318 ms scene rebuild; Apple M3 Pro 287–321 ms; fits the 0.4 s replan",
                  "Feature engineering: 21 hand-made properties (time to closest approach, threat, companion): no gain",
                  "Constant velocity baseline: Schöller et al. 2020 · TabPFN: Hollmann et al. 2025 (Nature)",
                  "Data: Pellegrini 2009, Lerner 2007 · calibration: Gibbs & Candès 2021"],
         notes="10 Hz is not claimed: it needs a background scene rebuild that is designed, not built. Full reference list: docs/VIDEO_BRIEF.md section 4."),
    dict(q="One decision in detail", pics=["docs/walkthrough/s5_person_box.png", "docs/walkthrough/s8_planner.png"],
         a="One real moment, every stage, with the actual values", notes="Seed 6001, t = 7.2 s, person 11. Full page: docs/walkthrough.html."),
    dict(q="Five real scenes, scene by scene", pics=["media/fig_loso5.png"],
         a="TabPFN on 1,500 rows: ahead of LightGBM on the same rows in 4 scenes, level on Hotel; level with LightGBM on about 22,000",
         notes="Interval score at +2 s, mean over 3 context seeds. TabPFN 1.50 / 1.06 / 1.72 / 1.28 / 1.23 (ETH, Hotel, Univ, Zara1, Zara2); constant velocity plus band 1.69 / 1.34 / 2.33 / 1.54 / 1.72. Zara1 and Zara2 share one camera."),
    dict(q="Is TabPFN only the forecaster?", clip="policy_seed3403", pics=["media/diagrams/02_pipeline.png"],
         a="A second TabPFN can pick the robot's move: on 20 new crowds it drove like the planner it learned from, on average",
         notes="Ranges, scene adaptation and the move are all TabPFN calls. Tracker, calibration offset and the hard safety check stay classical. Collision interval is wide at 20 crowds: don't say 'as safe'."),
]


def backups(d):
    # a divider, so one click too many after the closing slide shows no new material
    s = d.slide(footer=False, meta="Divider before the backup slides",
                notes="Nothing to say here. If you land on this slide by mistake, go back one. The backup slides follow, for questions only.")
    text(s, M, 3.0, CW, 0.8, "Backup slides", 40, bold=True)
    text(s, M, 3.9, CW, 0.4, "For questions", 18, color=MUTED)
    for i, b in enumerate(BACKUPS, 1):
        s = d.slide(f"Backup · B{i}", b["q"], notes=b.get("notes", ""), meta=f"Backup B{i}")
        top, bottom = 1.65, 6.05
        if b.get("bullets"):
            box(s, M, top, CW, bottom - top + 0.6)
            for k, t in enumerate(b["bullets"]):
                yy = top + 0.45 + k * 1.05
                box(s, M + 0.4, yy + 0.1, 0.14, 0.14, fill=ACCENT, line=None, kind=MSO_SHAPE.OVAL)
                text(s, M + 0.8, yy - 0.08, CW - 1.3, 0.9, t, 18)
            continue
        box(s, M, top, CW, bottom - top)
        items = ([("clip", b["clip"])] if b.get("clip") else []) + [("pic", p) for p in b.get("pics", [])]
        gap, pad = 0.3, 0.15
        iw = (CW - 2 * pad - gap * (len(items) - 1)) / len(items)
        for k, (kind, p) in enumerate(items):
            xx = M + pad + k * (iw + gap)
            rel = (DK + p + "_last.png" if STATIC else G + p + ".gif") if kind == "clip" else p
            pic(s, rel, xx, top + pad, iw, bottom - top - 2 * pad)
        text(s, M, 6.25, CW, 0.6, b["a"], 16, bold=True)


def build():
    d = Deck()
    talk(d)
    backups(d)
    out = ROOT / ("deck/Foresight_finalist_static.pptx" if STATIC else "deck/Foresight_finalist.pptx")
    out.parent.mkdir(exist_ok=True)
    d.prs.save(out)
    print(f"wrote {out} ({len(d.prs.slides)} slides)")


if __name__ == "__main__":
    build()
