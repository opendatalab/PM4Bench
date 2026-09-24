"""Shared offline browser/font setup for synthesis recipes."""
from __future__ import annotations

import json
from pathlib import Path

from ..data.benchmark import sha256
from ..io import resolve_asset
from .resources import FONTS


def new_output(output: Path, *inputs: Path) -> Path:
    output = output.resolve()
    for source in inputs:
        root = source.resolve()
        if output == root or root in output.parents:
            raise ValueError("Output must be outside input directories")
    output.mkdir(parents=True, exist_ok=False)
    return output


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font_css(fonts: Path, segoe: Path | None = None) -> tuple[str, dict]:
    css, hashes = [], {}
    for filename in sorted(set(FONTS.values())):
        path = resolve_asset(fonts, filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        # The public downloader includes variable files; local historical files
        # may be static despite sharing the same filename.
        from fontTools.ttLib import TTFont

        with TTFont(path) as font:
            axes = {axis.axisTag: axis for axis in font['fvar'].axes} if 'fvar' in font else {}
            axis = axes.get('wght')
            weights = f'{axis.minValue:g} {axis.maxValue:g}' if axis else '400'
        family = filename.removesuffix('-Regular.ttf')
        css.append(f'@font-face{{font-family:{family};src:url("{path.as_uri()}");'
                   f'font-weight:{weights};}}')
        hashes[filename] = sha256(path)
    if segoe:
        for face, weight in [('light', 200), ('regular', 400), ('bold', 700)]:
            path = resolve_asset(segoe, f'segoeui-{face}.woff2')
            if not path.is_file():
                raise FileNotFoundError(path)
            css.append(f'@font-face{{font-family:ReferenceSegoe;src:url("{path.as_uri()}");'
                       f'font-weight:{weight};}}')
            hashes[path.name] = sha256(path)
    return '\n'.join(css), hashes


def open_browser(playwright, executable: Path | None, *, no_hinting: bool = False):
    return playwright.chromium.launch(
        executable_path=str(executable) if executable else None,
        headless=True, args=['--font-render-hinting=none'] if no_hinting else [],
    )


def offline_page(browser, width: int, height: int):
    page = browser.new_page(viewport={'width': width, 'height': height},
                            device_scale_factor=1, service_workers='block')
    page.route('http://**/*', lambda route: route.abort())
    page.route('https://**/*', lambda route: route.abort())
    return page


def wait_for_render(page):
    """Resolve fallback faces before measuring visibility or taking a screenshot."""
    page.evaluate('''async () => {
      const elements = [...document.querySelectorAll('.word-item,p,li,strong')];
      await Promise.all(elements.map(el => {
        const style = getComputedStyle(el);
        return document.fonts.load(style.font, el.textContent);
      }));
      await document.fonts.ready;
      await Promise.all([...document.images].map(img => img.decode().catch(() => {})));
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }''')
