"""Export the inline SVG diagrams of docs/explainer.html to PNG for slides and video.

For each <figure> with an inline <svg>, writes a standalone HTML (the page's own
<style> block, light theme forced through the page's data-theme switch) into
media/diagrams/_tmp/, screenshots it with headless Edge or Chrome at device scale 2,
crops the margins and saves media/diagrams/NN_slug.png. Four key diagrams are also
exported with the dark tokens as NN_slug-dark.png. _tmp/ is deleted at the end.

Usage:  python -B tools/export_figures.py [--rev GITREV] [name-prefix ...]   (e.g. --rev HEAD 01 09)
The slug list below matches the 12-figure layout of commit 70372d5; a layout with a
different figure count stops with an error instead of mislabelling files.
Fonts come from Google Fonts (network); without it text falls back to Segoe UI.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "explainer.html"
OUT = ROOT / "media" / "diagrams"
TMP = OUT / "_tmp"

SLUGS = ["problem", "pipeline", "row", "tabpfn_step", "live_context", "planner",
         "assumptions", "coverage", "context_size", "t1_curve", "tests", "deployment"]
DARK = {"pipeline", "tabpfn_step", "planner", "assumptions"}

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

PAD = 80          # CSS px body padding (room for labels that spill past the viewBox)
SLACK = 120       # extra window height (CSS px); cropped away afterwards
OUT_PAD = 48      # device px of margin kept around the cropped diagram
BG = {"light": (255, 255, 255), "dark": (0x15, 0x1C, 0x25)}  # var(--surface)


class FigureParser(HTMLParser):
    """Collect each <figure>: raw svg source (by offsets), caption text, stray content."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.text = text
        self.lines = [0]
        for m in re.finditer("\n", text):
            self.lines.append(m.end())
        self.figs: list[dict] = []
        self.cur: dict | None = None
        self.svg_depth = 0
        self.in_cap = False

    def _off(self) -> int:
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag == "figure":
            self.cur = {"svg": None, "caption": "", "stray": [], "svg_start": None}
        elif self.cur is None:
            return
        elif tag == "svg":
            if self.svg_depth == 0:
                self.cur["svg_start"] = self._off()
            self.svg_depth += 1
        elif self.svg_depth:
            return
        elif tag == "figcaption":
            self.in_cap = True
        elif not self.in_cap and not (tag == "div" and ("class", "scroll") in attrs):
            self.cur["stray"].append(tag)

    def handle_endtag(self, tag):
        if self.cur is None:
            return
        if tag == "svg" and self.svg_depth:
            self.svg_depth -= 1
            if self.svg_depth == 0:
                end = self.text.index("</svg>", self._off()) + len("</svg>")
                self.cur["svg"] = self.text[self.cur["svg_start"]:end]
        elif tag == "figcaption":
            self.in_cap = False
        elif tag == "figure":
            self.figs.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cur is None or self.svg_depth:
            return
        if self.in_cap:
            self.cur["caption"] += data
        elif data.strip():
            self.cur["stray"].append(repr(data.strip()[:40]))


def strip_media_block(css: str, needle: str) -> str:
    """Remove an @media block (brace-matched) whose header contains needle."""
    i = css.find(needle)
    if i < 0:
        return css
    start = css.rfind("@media", 0, i)
    depth, j = 0, css.index("{", start)
    while True:
        if css[j] == "{":
            depth += 1
        elif css[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return css[:start] + css[j + 1:]


def build_html(style: str, font_links: str, svg: str, theme: str, w: int, h: int) -> str:
    css = strip_media_block(style, "prefers-color-scheme")
    override = (
        f":root{{color-scheme:{theme}}}\n"
        f"html,body{{margin:0;background:var(--surface)}}\n"
        f"body{{padding:{PAD}px}}\n"
        f"svg{{display:block;overflow:visible;width:{w}px;height:{h}px;color:var(--fg);font-family:var(--body)}}\n"
    )
    return (f'<!doctype html>\n<html data-theme="{theme}">\n<head>\n<meta charset="utf-8">\n'
            f"{font_links}\n<style>\n{css}\n{override}</style>\n</head>\n<body>\n{svg}\n</body>\n</html>\n")


def check_refs(svg: str) -> list[str]:
    ids = set(re.findall(r'\bid="([^"]+)"', svg))
    refs = set(re.findall(r'url\(#([^)]+)\)', svg)) | set(re.findall(r'href="#([^"]+)"', svg))
    return sorted(refs - ids)


TIMEOUT = 180     # s per screenshot (normally ~1 s; generous for a loaded machine)


def screenshot(browser: str, html: Path, png: Path, w: int, h: int) -> bool:
    """Render into _tmp first and move into place only on success, so a failed run
    never deletes an existing PNG."""
    shot = TMP / png.name
    if shot.exists():
        shot.unlink()
    cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--force-device-scale-factor=2", f"--window-size={w},{h}",
           "--no-first-run", "--no-default-browser-check", "--disable-extensions",
           "--disable-background-networking", "--disable-sync", "--mute-audio",
           f"--user-data-dir={TMP / 'profile'}", "--virtual-time-budget=4000",
           f"--screenshot={shot}", html.as_uri()]
    # Normal priority on purpose: one screenshot costs about a second of CPU, but at
    # below-normal priority the browser is starved for minutes when experiments hold 100% CPU.
    proc = subprocess.Popen(cmd, cwd=TMP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        # kill the whole browser tree, not just the launcher
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return False
    if shot.exists() and shot.stat().st_size > 0:
        shutil.move(str(shot), str(png))
        return True
    return False


def postprocess(png: Path, theme: str, w: int, h: int) -> tuple[str, tuple[int, int]]:
    """Check not blank and not clipped, crop to content plus margin. Returns (note, size)."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return "PIL missing: not verified or cropped", (0, 0)
    im = Image.open(png).convert("RGB")
    if all(a == b for a, b in im.getextrema()):
        return "BLANK (single colour)", im.size
    bg = Image.new("RGB", im.size, BG[theme])
    diff = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 8 else 0)
    box = diff.getbbox()
    if not box:
        return "BLANK (background only)", im.size
    W, H = im.size
    l, t, r, b = box
    notes = []
    # Expected svg rect in device px (scale 2). Content must sit inside it, and the
    # slack strip below it must be empty, which proves the viewport was not clipped.
    sx0, sy0, sx1, sy1 = 2 * PAD, 2 * PAD, 2 * (PAD + w), 2 * (PAD + h)
    if W < sx1 or H < sy1 + 20:
        notes.append(f"WARNING screenshot {W}x{H} smaller than svg box")
    over = max(sx0 - l, sy0 - t, r - sx1, b - sy1)
    if over > 4:
        # The page clips the svg at its viewBox; here overflow is drawn so labels stay whole.
        notes.append(f"note: source overflows viewBox by {over // 2} CSS px (drawn, page clips it)")
    if l <= 2 or t <= 2 or r >= W - 2 or b >= H - 2:
        notes.append("WARNING content touches screenshot edge (clipped)")
    out_w, out_h = r - l + 2 * OUT_PAD, b - t + 2 * OUT_PAD
    crop = im.crop(box)
    out = Image.new("RGB", (out_w, out_h), BG[theme])
    out.paste(crop, (OUT_PAD, OUT_PAD))
    out.save(png, optimize=True)
    return ("; ".join(notes) or "ok"), out.size


def rmtree_retry(p: Path) -> bool:
    for _ in range(10):
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    rev = None
    if "--rev" in args:   # read the explainer from a git revision (read-only), e.g. --rev HEAD
        k = args.index("--rev")
        rev = args[k + 1]
        del args[k:k + 2]
    only = args           # optional name prefixes, e.g. 01_problem 09 (default: all)
    if rev:
        text = subprocess.run(["git", "show", f"{rev}:docs/explainer.html"], cwd=ROOT,
                              capture_output=True, check=True).stdout.decode("utf-8")
    else:
        text = SRC.read_text(encoding="utf-8")
    print(f"source: docs/explainer.html @ {rev or 'working tree'}")
    style = re.search(r"<style>(.*?)</style>", text, re.S).group(1)
    font_links = "\n".join(re.findall(r'<link[^>]+fonts\.(?:googleapis|gstatic)[^>]*>', text))
    p = FigureParser(text)
    p.feed(text)
    figs = [f for f in p.figs if f["svg"]]
    if len(figs) != len(SLUGS):
        print(f"expected {len(SLUGS)} svg figures, found {len(figs)}", file=sys.stderr)
        return 1

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        print("no Edge or Chrome found", file=sys.stderr)
        return 1
    TMP.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, (f, slug) in enumerate(zip(figs, SLUGS), 1):
        vb = re.search(r'viewBox="([\d.\s-]+)"', f["svg"]).group(1).split()
        vw, vh = float(vb[2]), float(vb[3])
        # 1600 px for wide figures; small or portrait figures at 1000 px so text stays legible.
        w = 1600 if (vw >= 600 and vw >= vh) else 1000
        h = round(w * vh / vw)
        if f["stray"]:
            print(f"  WARNING {slug}: non-svg figure content dropped: {f['stray']}")
        missing = check_refs(f["svg"])
        if missing:
            print(f"  WARNING {slug}: unresolved refs {missing}")
        cap = " ".join(f["caption"].split())
        for theme in ["light"] + (["dark"] if slug in DARK else []):
            name = f"{i:02d}_{slug}" + ("-dark" if theme == "dark" else "")
            if only and not any(name.startswith(o) for o in only):
                continue
            jobs.append((name, theme, f["svg"], w, h, vw / vh, cap))

    used = browser
    results = []
    for name, theme, svg, w, h, aspect, cap in jobs:
        html = TMP / f"{name}.html"
        html.write_text(build_html(style, font_links, svg, theme, w, h), encoding="utf-8")
        png = OUT / f"{name}.png"
        win = (w + 2 * PAD, h + 2 * PAD + SLACK)
        ok = screenshot(used, html, png, *win)
        if not ok:
            alt = next((b for b in BROWSERS if b != used and Path(b).exists()), None)
            if alt and screenshot(alt, html, png, *win):
                used, ok = alt, True
        if not ok:
            results.append((name, "FAILED (no PNG)", (0, 0), cap))
            print(f"{name}.png  FAILED (no PNG)")
            continue
        note, size = postprocess(png, theme, w, h)
        results.append((name, note, size, cap))
        print(f"{name}.png  {size[0]}x{size[1]}  {note}", flush=True)

    print(f"\nbrowser: {used}")
    print("\ncaptions:")
    for name, _, _, cap in results:
        if not name.endswith("-dark"):
            print(f"  {name}: {cap}")
    gone = rmtree_retry(TMP)
    print(f"\n_tmp removed: {gone}")
    return 0 if all(not r[1].startswith(("FAILED", "BLANK")) for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
