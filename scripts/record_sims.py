#!/usr/bin/env python3
"""
Record the learn page's own simulations to video.

    python3 scripts/record_sims.py
    python3 scripts/record_sims.py --only envelope

These are screen recordings of the three browser simulations in
docs/learn.html, driven programmatically so the same demonstration comes out
every time. They are NOT footage of the robot - nothing here has run in Gazebo.
The page labels them accordingly, and the five robot slots stay empty until
someone records them with scripts/record_demo.sh.

Why record a simulation you can already drag? Two reasons. A reader skimming on
a phone will not drag anything, and a recording shows the intended sequence -
which slider, in which order, to make the point. The interactive version stays
underneath for anyone who wants to poke at it.

Output is WebM, which every current browser plays natively and which Playwright
produces directly, so there is no transcode step and no ffmpeg dependency.

A poster frame is captured for each clip by playing it back in the same browser
and screenshotting the video element. That is deliberate: a poster wants a frame
that shows the point being made, and picking one by seeking with a command-line
tool means trusting its seek accuracy on a VP8 stream with sparse keyframes.
Playing it in a real decoder and asking for the frame it is showing does not
have that problem. It also gives the reader something to look at before pressing
play, which matters because Playwright starts recording when the browser context
opens, so every clip begins with a moment of blank page.

Requires: pip install playwright  (the browser is already on this image)
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
PAGE = os.path.join(ROOT, 'docs', 'learn.html')
OUT = os.path.join(ROOT, 'docs', 'media')

CHROME_CANDIDATES = [
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/opt/pw-browsers/chromium/chrome-linux/chrome',
]

VIEW = {'width': 1000, 'height': 720}


def find_chrome() -> str | None:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    for pattern in ('/opt/pw-browsers/chromium-*/chrome-linux/chrome',):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


async def frame_the_panel(page) -> None:
    """Strip the page furniture so the recording is of the simulation alone.

    Left as-is, every frame carries the navigation rail and the page margins,
    so the panel occupies about half the width and the finished clip embedded
    back into this same page shows a small copy of the page inside itself. None
    of that is what the clip is meant to show. Hiding the rail and letting the
    column go full-bleed roughly doubles the pixels spent on the readouts,
    which are the part a viewer has to be able to read.

    This only affects the recording. Nobody browsing the page sees it.
    """
    await page.add_style_tag(content="""
        .rail { display: none !important; }
        body  { padding-left: 0 !important; }
        .wrap { max-width: none !important; padding: 0 14px 120px !important; }
        .sim  { margin: 0 !important; }
    """)


async def scroll_to(page, selector: str) -> None:
    """Frame the whole simulation panel, not just its canvas.

    Scrolling to the canvas puts the canvas at the top of the viewport, which
    pushes the panel heading off the top and the verdict line off the bottom -
    and the verdict is the sentence that says what the demonstration proved.
    Walk up to the enclosing .sim and frame that instead.
    """
    await page.eval_on_selector(
        selector,
        # Centring works only while the panel fits. Once it is taller than the
        # viewport, centring clips the heading off the top and the verdict off
        # the bottom - the two lines that say which simulation this is and what
        # it proved. In that case pin the top instead and accept losing the
        # tail, which is padding.
        "el => { const p = el.closest('.sim') || el;"
        "        const r = p.getBoundingClientRect();"
        "        const y = r.top + window.scrollY;"
        "        window.scrollTo({top: r.height >= innerHeight - 16"
        "                              ? y - 8"
        "                              : y - (innerHeight - r.height) / 2,"
        "                         behavior: 'instant'}); }")
    await page.wait_for_timeout(500)


async def slide(page, sel: str, value, hold: int = 700) -> None:
    """Move a range input the way a person would, then pause on the result."""
    await page.eval_on_selector(
        sel, f"el => {{ el.value = {value}; el.dispatchEvent(new Event('input')); }}")
    await page.wait_for_timeout(hold)


async def sweep(page, sel: str, start: int, end: int, steps: int = 26,
                dwell: int = 55) -> None:
    """Drag a slider smoothly, so the reader sees the change rather than a jump."""
    for i in range(steps + 1):
        value = round(start + (end - start) * i / steps)
        await slide(page, sel, value, hold=dwell)


# --- the three demonstrations ---------------------------------------------

async def demo_envelope(page):
    """Show reach failing as the riser grows, then the torque cost of fixing it."""
    await scroll_to(page, '#c-envelope')
    await page.wait_for_timeout(900)
    await sweep(page, '#s-riser', 150, 210)      # reach fails
    await page.wait_for_timeout(1400)
    await sweep(page, '#s-rclu', 115, 175)       # fix it, and watch torque climb
    await page.wait_for_timeout(1600)
    await slide(page, '#s-rclu', 115, 500)       # back to the shipped design
    await slide(page, '#s-riser', 150, 1500)


async def demo_tof(page):
    """Show a sill reading flat, a drop reading cliff, then the ramp bug."""
    await scroll_to(page, '#c-tof')
    await page.wait_for_timeout(900)
    await page.click('#b-preset-sill')           # 22 mm sill -> flat
    await page.wait_for_timeout(1500)
    await sweep(page, '#s-dz', 22, 150)          # into the riser band
    await page.wait_for_timeout(1300)
    await sweep(page, '#s-dz', 150, -120)        # over the edge -> cliff
    await page.wait_for_timeout(1300)
    await page.click('#b-preset-ramp')           # the pitch-correction case
    await page.wait_for_timeout(2400)


async def demo_climb(page):
    """Run the approach and let the stall trigger fire while pitch stays at zero."""
    await scroll_to(page, '#c-climb')
    await page.wait_for_timeout(800)
    await page.click('#b-climb-run')
    await page.wait_for_timeout(6500)            # long enough for the stall
    await page.click('#b-climb-reset')
    await slide(page, '#s-hold', 0, 600)         # then show the bob without the hold
    await page.click('#b-climb-run')
    await page.wait_for_timeout(5200)


DEMOS = {
    'sim-climb-envelope': (demo_envelope,
                           'Climb envelope: reach failing at a 210 mm riser, '
                           'and the torque cost of fixing it'),
    'sim-tof-classify': (demo_tof,
                         'The ToF beam: sill, riser, cliff, then the ramp that '
                         'needs pitch correction'),
    'sim-stall-trigger': (demo_climb,
                          'The approach: a stall trigger fires while body pitch '
                          'never leaves zero'),
}


async def record(name: str, fn, chrome: str) -> str:
    from playwright.async_api import async_playwright

    staging = os.path.join(OUT, f'.staging-{name}')
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=chrome)
        context = await browser.new_context(
            viewport=VIEW, record_video_dir=staging, record_video_size=VIEW,
            # The page's fonts come from a CDN. Recording offline would show a
            # flash of fallback text, so give it a moment either way.
            reduced_motion='no-preference')
        page = await context.new_page()
        await page.goto(f'file://{PAGE}')
        await frame_the_panel(page)
        await page.wait_for_timeout(1200)
        await fn(page)
        await page.wait_for_timeout(400)
        await context.close()
        await browser.close()

    produced = glob.glob(os.path.join(staging, '*.webm'))
    if not produced:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(f'{name}: playwright produced no video')

    target = os.path.join(OUT, f'{name}.webm')
    shutil.move(produced[0], target)
    shutil.rmtree(staging, ignore_errors=True)
    await capture_poster(target, chrome)
    return target


async def capture_poster(video_path: str, chrome: str,
                         at_fraction: float = 0.62) -> str | None:
    """Screenshot a frame from the finished clip, using the browser to decode.

    `at_fraction` lands past the middle, where these demonstrations have made
    their point, and comfortably clear of the blank lead-in at the start.
    """
    from playwright.async_api import async_playwright

    poster = video_path.rsplit('.', 1)[0] + '.png'
    # The holder has to live beside the clip and be opened as a file:// page of
    # its own. A page built with set_content is about:blank, and Chromium will
    # not let about:blank read a file:// video: the element simply never
    # reaches readyState 2, which reads exactly like a decode that hung.
    holder = os.path.join(os.path.dirname(video_path), '.poster-holder.html')
    with open(holder, 'w') as fh:
        fh.write('<style>html,body{margin:0;background:#0E1114}'
                 'video{display:block;width:100vw}</style>'
                 f'<video id="v" src="{os.path.basename(video_path)}" muted>'
                 '</video>')
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=chrome)
        page = await (await browser.new_context(viewport=VIEW)).new_page()
        await page.goto(f'file://{holder}')
        video = page.locator('#v')
        try:
            await page.wait_for_function(
                "() => { const v = document.getElementById('v');"
                "        return v.readyState >= 2 && isFinite(v.duration); }",
                timeout=15000)
            await page.evaluate(
                "f => { const v = document.getElementById('v');"
                "       v.currentTime = v.duration * f; }", at_fraction)
            await page.wait_for_function(
                "() => document.getElementById('v').readyState >= 2",
                timeout=10000)
            await page.wait_for_timeout(600)
            await video.screenshot(path=poster)
        except Exception as exc:                   # noqa: BLE001
            print(f'    (no poster: {exc})')
            poster = None
        await browser.close()
    os.path.exists(holder) and os.remove(holder)
    return poster


async def main_async(args) -> int:
    chrome = find_chrome()
    if chrome is None:
        print('No Chromium found under /opt/pw-browsers. On a normal machine:\n'
              '  pip install playwright && playwright install chromium',
              file=sys.stderr)
        return 1
    if not os.path.exists(PAGE):
        print(f'{PAGE} does not exist', file=sys.stderr)
        return 1

    os.makedirs(OUT, exist_ok=True)
    wanted = {k: v for k, v in DEMOS.items()
              if not args.only or args.only in k}
    if not wanted:
        print(f'no demo matching "{args.only}". Known: {", ".join(DEMOS)}',
              file=sys.stderr)
        return 1

    print(f'recording {len(wanted)} simulation(s) at '
          f'{VIEW["width"]}x{VIEW["height"]}\n')
    for name, (fn, description) in wanted.items():
        print(f'  {name} — {description}')
        path = await record(name, fn, chrome)
        size = os.path.getsize(path)
        flag = '  (over 3 MB, consider trimming)' if size > 3_145_728 else ''
        poster = path.rsplit('.', 1)[0] + '.png'
        extra = (f' + poster {os.path.getsize(poster) / 1024:.0f} KB'
                 if os.path.exists(poster) else ' (poster failed)')
        print(f'    → {os.path.relpath(path, ROOT)}  '
              f'{size / 1024:.0f} KB{flag}{extra}\n')

    print('These are recordings of the browser simulations, not of the robot.\n'
          'The five Gazebo slots stay empty until someone runs '
          'scripts/record_demo.sh.')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='record just the demos matching this substring')
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())
