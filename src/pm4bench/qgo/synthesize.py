"""Multilingual synthetic OCR images, visible-word targets, and replayable plans."""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from ..data.benchmark import sha256
from ..io import iter_jsonl, write_jsonl
from ..vision.browser import (
    font_css,
    new_output,
    offline_page,
    open_browser,
    save_json,
    wait_for_render,
)
from ..vision.resources import RESOURCE_ROOT

TEXT_POOL = Path(__file__).parent / 'resources/title_dict_sample.json'
LANGUAGES = ('en', 'ar', 'cs', 'ko', 'hu', 'ru', 'sr', 'th', 'vi', 'zh')
SHADOWS = ('0px 0px 3px red', '2px 2px 3px blue', 'none')


def load_words(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    words = {lang: [item[lang] for item in data.values() if lang in item] for lang in LANGUAGES}
    if any(any(not isinstance(v, str) for v in values) for values in words.values()):
        raise ValueError('Text pool values must be strings')
    # The historical pool contains 41 blank translations. Keep the bundled
    # source unchanged, but do not sample blank foreground words in new corpora.
    words = {lang: [v for v in values if v.strip()] for lang, values in words.items()}
    if any(not values for values in words.values()):
        raise ValueError('Text pool must provide nonempty text for all ten languages')
    return words


def sample_plan(words: dict, index: int, seed: int) -> dict:
    # Per-record random state makes the recipe independent of worker scheduling.
    rng = random.Random(f'{seed}:{index}')
    width, height = rng.choice((800, 1024, 1280, 1920)), rng.choice((600, 768, 1080, 1440))
    languages = list(LANGUAGES)
    rng.shuffle(languages)
    lines = []
    for i in range(height // 60):
        language = languages[i % len(languages)]
        items = []
        for _ in range(width // 80):
            items.append({'text': rng.choice(words[language]),
                          'size': rng.choice((14, 18, 22, 28, 32)),
                          'weight': rng.choice(('normal', 'bold', 'bolder')),
                          'style': rng.choice(('normal', 'italic', 'oblique')),
                          'shadow': rng.choice(SHADOWS)})
        lines.append({'language': language, 'words': items})
    return {'id': f'ocr_{index:06d}', 'width': width, 'height': height,
            'background': rng.randrange(1, 10), 'lines': lines, 'seed': seed, 'index': index}


def validate_plan(plan: dict) -> None:
    import re

    if not re.fullmatch(r'[A-Za-z0-9_-]+', plan['id']):
        raise ValueError('Invalid plan id')
    if not (128 <= plan['width'] <= 4096 and 128 <= plan['height'] <= 4096):
        raise ValueError('Invalid viewport size')
    if plan['background'] not in range(1, 10) or not plan['lines']:
        raise ValueError('Invalid background or empty plan')
    for line in plan['lines']:
        if line['language'] not in LANGUAGES or not line['words']:
            raise ValueError('Invalid language/words')
        for word in line['words']:
            if (word['size'] not in (14, 18, 22, 28, 32)
                    or word['weight'] not in ('normal', 'bold', 'bolder')
                    or word['style'] not in ('normal', 'italic', 'oblique')
                    or word['shadow'] not in SHADOWS
                    or not isinstance(word['text'], str) or not word['text'].strip()):
                raise ValueError('Invalid word/style')


def make_html(plan: dict, fonts: str) -> str:
    validate_plan(plan)
    background = (RESOURCE_ROOT / f"backgrounds/background{plan['background']}.jpg").as_uri()
    lines = []
    for line in plan['lines']:
        language = line['language']
        spans = []
        for word in line['words']:
            style = (f"font-size:{word['size']}px;font-weight:{word['weight']};"
                     f"font-style:{word['style']};text-shadow:{word['shadow']}")
            spans.append(f'<span class="word-item" data-lang="{language}" style="{style}">'
                         f'{html.escape(word["text"])}</span>')
        direction = 'rtl' if language == 'ar' else 'ltr'
        lines.append(f'<div class="line-row" dir="{direction}">{"".join(spans)}</div>')
    return f'''<!doctype html><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="script-src 'none'; connect-src 'none'">
<style>{fonts}
body{{margin:0;padding:0;background:#f0f0f0 url("{background}") center/cover;
font-family:NotoSansArabic,NotoSansSC,NotoSansKR,NotoSansThai,NotoSans,sans-serif;
width:{plan['width']}px;height:{plan['height']}px;overflow:hidden;display:flex;
flex-direction:column;justify-content:center;align-items:center;box-sizing:border-box;}}
.main-container{{width:90%;height:90%;display:flex;flex-direction:column;
justify-content:space-around;align-items:stretch;gap:10px;}}
.line-row{{display:flex;flex-wrap:nowrap;align-items:center;padding:5px;}}
.word-item{{line-height:1.5;padding:0 10px;color:#000;white-space:nowrap;}}
</style><div class="main-container">{''.join(lines)}</div>'''


def sort_boxes(boxes: list[dict]) -> list[dict]:
    """Historical vertical-center grouping and Arabic right-to-left ordering."""
    if not boxes:
        return []
    threshold = sum(b['height'] for b in boxes) / len(boxes) * .5
    lines = []
    for box in sorted(boxes, key=lambda b: b['y']):
        center = box['y'] + box['height'] / 2
        if not lines or abs(center - (lines[-1][0]['y'] + lines[-1][0]['height'] / 2)) >= threshold:
            lines.append([])
        lines[-1].append(box)
    return [box for line in lines for box in sorted(
        line, key=lambda b: b['x'], reverse=line[0]['lang'] == 'ar')]


VISIBLE_WORDS = '''() => [...document.querySelectorAll('.word-item')].flatMap(el => {
 const r=el.getBoundingClientRect();
 if(r.top<2 || r.left<2 || r.bottom>innerHeight-2 || r.right>innerWidth-2 ||
    r.width<=0 || r.height<=0) {el.style.visibility='hidden';return [];}
 return [{text:el.innerText,x:r.x,y:r.y,width:r.width,height:r.height,
          lang:el.getAttribute('data-lang')}];
})'''


def synthesize(output: Path, fonts_root: Path, *, count: int = 20000, seed: int = 42,
               text_pool: Path = TEXT_POOL, plans_path: Path | None = None,
               browser_executable: Path | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    if count < 1:
        raise ValueError('Count must be positive')
    fonts, hashes = font_css(fonts_root)
    if plans_path:
        plans = list(iter_jsonl(plans_path))
    else:
        words = load_words(text_pool)
        plans = [sample_plan(words, i, seed) for i in range(count)]
    if not plans or len({p['id'] for p in plans}) != len(plans):
        raise ValueError('Empty/duplicate plans')
    for plan in plans:
        validate_plan(plan)
    inputs = [fonts_root, text_pool.parent]
    if plans_path:
        inputs.append(plans_path.parent)
    output = new_output(output, *inputs)
    (output / 'img').mkdir()
    (output / 'gt').mkdir()
    (output / 'html').mkdir()
    write_jsonl(output / 'plans.jsonl', plans)
    rows = []
    with sync_playwright() as playwright:
        browser = open_browser(playwright, browser_executable)
        page = offline_page(browser, 800, 600)
        for plan in plans:
            page.set_viewport_size({'width': plan['width'], 'height': plan['height']})
            path = output / 'html' / f"{plan['id']}.html"
            path.write_text(make_html(plan, fonts), encoding='utf-8')
            page.goto(path.as_uri())
            wait_for_render(page)
            boxes = sort_boxes(page.evaluate(VISIBLE_WORDS))
            if not boxes:
                raise ValueError(f"{plan['id']}: no visible words")
            image = output / 'img' / f"{plan['id']}.png"
            page.screenshot(path=str(image))
            target = ' '.join(box['text'] for box in boxes)
            save_json(output / 'gt' / f"{plan['id']}.json", {
                'image_file': image.name, 'width': plan['width'], 'height': plan['height'],
                'language': 'multi', 'gt_text': target, 'word_boxes': boxes,
            })
            rows.append({'id': plan['id'], 'image': f'img/{image.name}', 'sha256': sha256(image)})
        version = browser.version
        browser.close()
    write_jsonl(output / 'images.jsonl', rows)
    report = {'records': len(rows), 'browser': version, 'fonts_sha256': hashes,
              'seed': seed, 'replayed_plans': bool(plans_path),
              'plans_sha256': sha256(output / 'plans.jsonl'),
              'text_pool_sha256': None if plans_path else sha256(text_pool),
              'backgrounds_sha256': {p.name: sha256(p)
                                     for p in sorted((RESOURCE_ROOT / 'backgrounds').glob('*.jpg'))}}
    save_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--fonts-root', type=Path, required=True)
    parser.add_argument('--text-pool', type=Path, default=TEXT_POOL)
    parser.add_argument('--count', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plans', type=Path)
    parser.add_argument('--browser-executable', type=Path)
    args = parser.parse_args()
    print(json.dumps(synthesize(args.output_root, args.fonts_root, count=args.count, seed=args.seed,
                               text_pool=args.text_pool, plans_path=args.plans,
                               browser_executable=args.browser_executable), indent=2))


if __name__ == '__main__':
    main()
