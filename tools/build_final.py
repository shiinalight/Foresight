"""Refresh the generated copies in final/ from their sources.

    python tools/build_final.py

final/ is the short reference set: the few pages needed to answer a question about Foresight without searching
the working docs. `00_START_HERE.md` and `CTO_CHANGES.md` are written by hand. The numbered pages below are
copies: edit the source, then rerun this script. Relative links are rewritten so they still open from final/.
"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final"
COPIES = [("docs/STORY.md", "01_STORY.md"),
          ("docs/VIDEO_BRIEF.md", "02_VIDEO.md"),
          ("docs/DECK_OUTLINE.md", "03_DECK.md"),
          ("docs/HIGHWAY.md", "04_RESULTS_CARS.md"),
          ("docs/LOSO5.md", "05_RESULTS_FIVE_SCENES.md"),
          ("docs/DEFENSE.md", "06_QUESTIONS_AND_ANSWERS.md")]
LINK = re.compile(r"\]\(([^)\s#]+)(#[^)\s]*)?\)")


def relink(text, src):
    def fix(m):
        target, anchor = m.group(1), m.group(2) or ""
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):                 # http:, mailto: and the like
            return m.group(0)
        new = os.path.relpath((src.parent / target).resolve(), FINAL).replace("\\", "/")
        return f"]({new}{anchor})"
    return LINK.sub(fix, text)


def main():
    FINAL.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for src_rel, name in COPIES:
        src = ROOT / src_rel
        body = relink(src.read_text(encoding="utf-8"), src)
        head = (f"> **Copy of [`{src_rel}`](../{src_rel})**, made by `tools/build_final.py` on {stamp} "
                f"(after commit `{commit}`). Edit the source and rerun the script; do not edit this copy.\n\n")
        (FINAL / name).write_text(head + body, encoding="utf-8")
        print(f"{src_rel} -> final/{name}")


if __name__ == "__main__":
    main()
