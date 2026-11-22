"""Record the 3D page as an MP4, frame by frame (exact timing, no screen recorder needed).

    python tools/record_clip.py --intro                       # results/clip_intro.mp4 (how the prediction works)
    python tools/record_clip.py --outro                       # results/clip_outro.mp4 (the evidence)
    python tools/record_clip.py --scene --view auto           # results/clip_scene.mp4 (the street with predictions)
    python tools/record_clip.py --stitch                      # results/demo_full.mp4 = intro + scene + outro
    python tools/record_clip.py --video --stills 20 60 70 120 # video mode: still frames first, to check
    python tools/record_clip.py --video --fps 30              # results/demo_video.mp4, the final video (one timeline)

Uses Playwright driving the installed Google Chrome. Every frame is set explicitly through the page's
__introFrame / __sceneFrame hooks, then screenshotted, so the clip is smooth however slow the render.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright


def record(page_url, out, frame_js, seconds, fps, size):
    w, h = size
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        # headless, but on the Mac's GPU through Metal (no window to close; much faster than software rendering)
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--hide-scrollbars", "--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": w, "height": h})
        page.goto(page_url)
        page.wait_for_function("typeof window.__introFrame === 'function'", timeout=60000)
        page.wait_for_function("window.__assetsReady !== false", timeout=60000)   # the proofs' figures and clip frames
        page.wait_for_timeout(1500)                                  # textures and fonts
        if seconds is None:
            seconds = page.evaluate("window.__videoLength")
        n = int(round(seconds * fps))
        t0 = time.time()
        for i in range(n):
            page.evaluate(f"{frame_js}({i / fps})")
            page.screenshot(path=f"{tmp}/f{i:05d}.png")
            if i % (fps * 5) == 0:
                print(f"  {i / fps:5.1f} s / {seconds:.0f} s  ({time.time() - t0:.0f} s elapsed)", flush=True)
        browser.close()
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/f%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(f"wrote {out} ({n} frames, {seconds:.0f} s, {w}x{h})")


def stills(page_url, times, fps, size):
    """One frame per time, with the 5 s before it simulated so camera easing matches the video."""
    w, h = size
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--hide-scrollbars", "--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": w, "height": h})
        page.goto(page_url)
        page.wait_for_function("typeof window.__videoWarm === 'function'", timeout=60000)
        page.wait_for_function("window.__assetsReady !== false", timeout=60000)
        page.wait_for_timeout(1500)
        print(f"video length {page.evaluate('window.__videoLength'):.1f} s")
        for T in times:
            page.evaluate(f"window.__videoWarm({T}, {fps})")
            out = Path(f"results/video_still_{T:g}.png"); page.screenshot(path=str(out)); print(f"wrote {out}")
        browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="results/street3d_pred_ep0.html")
    ap.add_argument("--intro", action="store_true")
    ap.add_argument("--outro", action="store_true")
    ap.add_argument("--scene", action="store_true")
    ap.add_argument("--view", default="auto", help="scene camera: auto (scripted), overview, follow, street")
    ap.add_argument("--stitch", action="store_true", help="join clip_intro + clip_scene + clip_outro")
    ap.add_argument("--video", action="store_true", help="video mode: intro + scenario + outro in one timeline")
    ap.add_argument("--stills", type=float, nargs="*", help="with --video: only these still frames (seconds)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, nargs=2, default=[1920, 1080])
    a = ap.parse_args()
    url = Path(a.page).resolve().as_uri()
    if a.video and a.stills:
        stills(url + "?video&pause", a.stills, a.fps, a.size)
    elif a.video:
        record(url + "?video&pause", Path("results/demo_video.mp4"), "window.__videoFrame", a.seconds, a.fps, a.size)
    if a.intro:
        record(url + "?intro&pause", Path("results/clip_intro.mp4"), "window.__introFrame", a.seconds or 50.0, a.fps, a.size)
    if a.outro:
        record(url + "?outro&pause", Path("results/clip_outro.mp4"), "window.__outroFrame", a.seconds or 48.0, a.fps, a.size)
    if a.scene:
        record(url + f"?view={a.view}&pause", Path("results/clip_scene.mp4"), "window.__sceneFrame", a.seconds or 60.0, a.fps, a.size)
    if a.stitch:
        parts = [Path(f"results/clip_{p}.mp4").resolve() for p in ("intro", "scene", "outro")]
        missing = [p.name for p in parts if not p.exists()]
        if missing:
            raise SystemExit(f"missing {missing}: record them first")
        lst = Path("results/clip_list.txt"); lst.write_text("".join(f"file '{p}'\n" for p in parts))
        out = Path("results/demo_full.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(out)], check=True)
        lst.unlink()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
