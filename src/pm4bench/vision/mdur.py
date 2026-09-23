"""MDUR synthesis from public transcripts, source images, and reference-led styles."""
from __future__ import annotations

import argparse
import html
import io
import json
import math
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

from ..io import iter_jsonl, resolve_asset, write_jsonl
from .browser import font_css, new_output, offline_page, open_browser, save_json, wait_for_render
from .inputs import FONTS, RESOURCE_ROOT, image_map, load_manifest, sha256, text_digest
from .mdur_text import parse_vision_text

SHADOWS = {'none': 'none', 'red': '0px 0px 2px red',
           'blue': '2px 2px 2px blue', 'green': '4px 4px 4px green'}
DEFAULT_STYLE = {'size': 32, 'weight': 100, 'style': 'normal',
                 'decoration': 'none', 'shadow': 'red', 'background': 2}


class SafeLegacyText(HTMLParser):
    """Keep inert inline markup and browser-like placeholder removal, no active HTML."""
    allowed = frozenset({'b', 'strong', 'i', 'em', 'sub', 'sup', 'br', 'u'})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        del attrs
        if tag in {'script', 'style', 'iframe', 'object'}:
            self.suppressed += 1
        elif not self.suppressed and tag in self.allowed:
            self.parts.append(f'<{tag}>')

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'iframe', 'object'}:
            self.suppressed = max(0, self.suppressed - 1)
        elif not self.suppressed and tag in self.allowed - {'br'}:
            self.parts.append(f'</{tag}>')

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(html.escape(data))


def safe_text(text: str, *, breaks: bool = False) -> str:
    parser = SafeLegacyText()
    parser.feed(text.replace('\n', '<br>') if breaks else text)
    parser.close()
    return ''.join(parser.parts)


@lru_cache(maxsize=1)
def routing_metadata() -> dict:
    return json.loads((RESOURCE_ROOT / 'mdur_image_routing.json').read_text(encoding='utf-8'))


def mdur_plan(row: dict, root: Path) -> dict:
    parsed = parse_vision_text(row)
    images = image_map(row, root)
    routing = routing_metadata()[row['language']][row['source_id']]
    if len(routing['option_images']) != len(parsed['options']):
        raise ValueError(f"{row['id']}: option/image routing mismatch")
    used = set(routing['question_images']) | {
        i for indices in routing['option_images'] for i in indices}
    excluded = set(routing['explanation_images'])
    if used & excluded or used | excluded != set(images):
        raise ValueError(f"{row['id']}: incomplete or overlapping image routing")
    return {'id': row['id'], 'source_id': row['source_id'], 'language': row['language'],
            'task': 'mdur', **parsed, 'rendered_text_sha256': text_digest(row['rendered_text']),
            'question_images': [images[i] for i in routing['question_images']],
            'option_images': [[images[i] for i in group] for group in routing['option_images']],
            'excluded_explanation_images': [images[i] for i in sorted(excluded)],
            'reference_image': row['vision_image']}


def validate_style(style: dict):
    if (set(style) != set(DEFAULT_STYLE) or style['size'] not in range(14, 61)
            or style['weight'] not in (100, 200, 300, 400, 600, 700, 900)
            or style['style'] not in ('normal', 'italic', 'oblique')
            or style['decoration'] not in ('none', 'underline')
            or style['shadow'] not in SHADOWS or style['background'] not in range(1, 10)):
        raise ValueError('Invalid MDUR style record')


def make_html(plan: dict, root: Path, fonts: str, segoe: bool) -> str:
    def image_tag(path, width):
        uri = resolve_asset(root, path).as_uri()
        return f'<img src="{html.escape(uri, quote=True)}" width="{width}">'

    options = []
    for i, (text, images) in enumerate(zip(plan['options'], plan['option_images'])):
        content = f'({chr(65+i)})&nbsp; {safe_text(text)}'
        options.append(f'<li class="col-6">{content} '
                       + ' '.join(image_tag(p, 640) for p in images) + '</li>')
    family = FONTS[plan['language']].removesuffix('-Regular.ttf')
    if segoe:
        family = 'ReferenceSegoe,' + family
    bootstrap = (RESOURCE_ROOT / 'bootstrap.min.css').as_uri()
    return f'''<!doctype html><html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="script-src 'none'; connect-src 'none'">
<link href="{bootstrap}" rel="stylesheet"><style>{fonts}
body{{padding:20px;padding-top:100px;font-family:{family},sans-serif;line-height:1.5;
background-size:cover;background-position:top center;background-repeat:no-repeat;
background-origin:padding-box;background-clip:padding-box;}}
.question-container{{padding:20px;margin-bottom:20px;background-color:inherit;border-radius:5px;}}
.options{{list-style-type:none;padding-left:0;display:flex;flex-direction:column;gap:10px;}}
.options li{{flex:1 1 100%;display:flex;align-items:center;}}
</style></head><body><div class="container mt-5"><div class="question-container">
<p><strong>{html.escape(plan['question_label'])}:&nbsp;</strong>
{safe_text(plan['question'], breaks=True)}</p>
{' '.join(image_tag(p, 700) for p in plan['question_images'])}
<p><strong>{html.escape(plan['option_label'])}:</strong></p>
<ul class="options row">{''.join(options)}</ul></div></div></body></html>'''


def screenshot(page, style: dict) -> bytes:
    validate_style(style)
    background = (RESOURCE_ROOT / f"backgrounds/background{style['background']}.jpg").as_uri()
    page.set_viewport_size({'width': 1280, 'height': 200})
    page.evaluate('''s => Object.assign(document.body.style, {
        fontSize:s.size+'px', fontWeight:String(s.weight), fontStyle:s.style,
        textDecoration:s.decoration, textShadow:s.textShadow,
        backgroundImage:'url("'+s.backgroundUrl+'")'})''',
        {**style, 'textShadow': SHADOWS[style['shadow']], 'backgroundUrl': background})
    wait_for_render(page)
    height = page.evaluate('''() => Math.max(document.documentElement.scrollHeight,
                                            document.body.getBoundingClientRect().height)''')
    if not 200 <= height <= 30000:
        raise ValueError(f'Unexpected MDUR image height: {height}')
    page.set_viewport_size({'width': 1280, 'height': math.ceil(height)})
    return page.screenshot()


def fit_background(reference) -> tuple[int, float]:
    import numpy as np
    from PIL import Image

    height, width = reference.shape[:2]
    ys, xs = np.meshgrid(np.linspace(0, height-2, 160).astype(int),
                         np.array([0, 10, 25, 40, 60, width-15]), indexing='ij')
    observed = reference[ys, xs].astype(float)
    scores = []
    for index in range(1, 10):
        with Image.open(RESOURCE_ROOT / f'backgrounds/background{index}.jpg') as image:
            pixels = np.asarray(image.convert('RGB'))
        h, w = pixels.shape[:2]
        scale = max(width/w, (height-1)/h)
        bx = np.clip(((xs + (w*scale-width)/2)/scale).astype(int), 0, w-1)
        by = np.clip((ys/scale).astype(int), 0, h-1)
        scores.append((float(np.abs(observed-pixels[by, bx].astype(float)).mean()), index))
    scores.sort()
    return scores[0][1], scores[1][0]-scores[0][0]


def reference_score(reference, content: bytes) -> float:
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(content)) as image:
        candidate = np.asarray(image.convert('RGB'))
    height = min(reference.shape[0], candidate.shape[0])
    # Score actual renders, never copy reference pixels into the output.
    error = np.abs(reference[:height, 90:1190].astype(float)
                   - candidate[:height, 90:1190].astype(float)).mean()
    return float(error + 30 * abs(len(reference)-len(candidate))/len(reference))


def choose_style(page, reference, initial: dict) -> tuple[dict, bytes, float]:
    """Bounded coordinate search; a measured approximation, not recovered metadata."""
    best = dict(initial)
    content = screenshot(page, best)
    score = reference_score(reference, content)
    candidates = [dict(best, size=size, weight=weight)
                  for size in (26, 28, 30, 32, 34, 36) for weight in (100, 400, 700, 900)]
    for stage in ('geometry', 'style', 'shadow', 'decoration'):
        if stage != 'geometry':
            values = {'style': ('normal', 'italic', 'oblique'),
                      'shadow': tuple(SHADOWS), 'decoration': ('none', 'underline')}[stage]
            candidates = [dict(best, **{stage: value}) for value in values]
        for candidate in candidates:
            if candidate == best:
                continue
            rendered = screenshot(page, candidate)
            value = reference_score(reference, rendered)
            if value < score:
                best, content, score = candidate, rendered, value
    return best, content, score


def render_mdur(dataset_root: Path, output_root: Path, languages: tuple[str, ...], *,
                fonts_root: Path | None = None, segoe_root: Path | None = None,
                ids: tuple[str, ...] | None = None, limit: int | None = None,
                audit_only: bool = False, fit: bool = True, styles: Path | None = None,
                browser_executable: Path | None = None) -> dict:
    root = dataset_root.resolve()
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    plans, manifests = [], {}
    for language in languages:
        rows, digest = load_manifest(root, 'mdur', language)
        selected = [row for row in rows if ids is None or row['id'] in ids]
        plans.extend(mdur_plan(row, root) for row in (selected[:limit] if limit else selected))
        manifests[language] = digest
    if not plans or (ids and set(ids) - {p['id'] for p in plans}):
        raise ValueError('No records selected, or requested ids not found')
    if not audit_only and fonts_root is None:
        raise ValueError('Rendering requires fonts_root')
    fonts, hashes = ('', {}) if audit_only else font_css(fonts_root, segoe_root)
    supplied = {r['id']: r['style'] for r in iter_jsonl(styles)} if styles else {}
    if styles and set(supplied) != {p['id'] for p in plans}:
        raise ValueError('Style manifest must cover exactly the selected records')
    for style in supplied.values():
        validate_style(style)
    inputs = [root] + [p for p in (fonts_root, segoe_root) if p is not None]
    output = new_output(output_root, *inputs)
    write_jsonl(output / 'plans.jsonl', plans)
    report = {'task': 'mdur', 'records': len(plans), 'audit_only': audit_only,
              'input_manifests_sha256': manifests, 'fonts_sha256': hashes,
              'font_profile': 'segoe-with-noto-fallback' if segoe_root else 'noto',
              'style_selection': 'supplied' if styles else 'reference-fit' if fit else 'calibrated',
              'pixel_identical': False}
    artifacts, chosen = [], []
    if not audit_only:
        import numpy as np
        from PIL import Image
        from playwright.sync_api import sync_playwright

        (output / 'html').mkdir()
        (output / 'images').mkdir()
        with sync_playwright() as playwright:
            browser = open_browser(playwright, browser_executable, no_hinting=True)
            page = offline_page(browser, 1280, 200)
            for plan in plans:
                path = output / 'html' / f"{plan['id']}.html"
                path.write_text(make_html(plan, root, fonts, bool(segoe_root)), encoding='utf-8')
                page.goto(path.as_uri())
                wait_for_render(page)
                reference_path = resolve_asset(root, plan['reference_image'])
                with Image.open(reference_path) as image:
                    reference = np.asarray(image.convert('RGB'))
                background, margin = fit_background(reference)
                style = supplied.get(plan['id'], dict(DEFAULT_STYLE, background=background))
                if fit and not styles:
                    style, content, score = choose_style(page, reference, style)
                else:
                    content = screenshot(page, style)
                    score = reference_score(reference, content)
                # Persist the winning style in HTML too, not the final search candidate.
                screenshot(page, style)
                path.write_text(page.content(), encoding='utf-8')
                destination = output / 'images' / f"{plan['id']}.png"
                destination.write_bytes(content)
                artifacts.append({'id': plan['id'], 'image': f'images/{destination.name}',
                                  'sha256': sha256(destination)})
                chosen.append({'id': plan['id'], 'style': style, 'reference_score': score,
                               'background_fit_margin': margin,
                               'reference_sha256': sha256(reference_path)})
            report['browser'] = browser.version
            browser.close()
        write_jsonl(output / 'images.jsonl', artifacts)
        write_jsonl(output / 'styles.jsonl', chosen)
    report['images'] = len(artifacts)
    save_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--fonts-root', type=Path)
    parser.add_argument('--segoe-root', type=Path)
    parser.add_argument('--language', action='append', choices=tuple(FONTS))
    parser.add_argument('--id', action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--no-fit', action='store_true')
    parser.add_argument('--styles', type=Path)
    parser.add_argument('--browser-executable', type=Path)
    args = parser.parse_args()
    if not args.audit_only and not args.fonts_root:
        parser.error('--fonts-root is required for rendering')
    print(json.dumps(render_mdur(
        args.dataset_root, args.output_root, tuple(args.language or FONTS),
        fonts_root=args.fonts_root, segoe_root=args.segoe_root,
        ids=tuple(args.id) if args.id else None, limit=args.limit, audit_only=args.audit_only,
        fit=not args.no_fit, styles=args.styles, browser_executable=args.browser_executable), indent=2))


if __name__ == '__main__':
    main()
