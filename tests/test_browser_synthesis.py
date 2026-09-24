"""Optional browser regressions: set PM4BENCH_TEST_FONTS to downloaded Noto fonts."""
import os
from pathlib import Path

import pytest

from pm4bench.qgo.synthesize import synthesize


@pytest.mark.skipif(not os.environ.get('PM4BENCH_TEST_FONTS'), reason='Needs browser and Noto fonts')
def test_multilingual_browser_replay(tmp_path):
    fonts = Path(os.environ['PM4BENCH_TEST_FONTS'])
    original, replay = tmp_path / 'original', tmp_path / 'replay'
    synthesize(original, fonts, count=12)
    synthesize(replay, fonts, plans_path=original / 'plans.jsonl')
    assert (original / 'images.jsonl').read_bytes() == (replay / 'images.jsonl').read_bytes()
    for path in (original / 'gt').glob('*.json'):
        assert path.read_bytes() == (replay / 'gt' / path.name).read_bytes()


@pytest.mark.skipif(not os.environ.get('PM4BENCH_TEST_FONTS'), reason='Needs browser')
@pytest.mark.parametrize('limit', [1, 2])
def test_mgui_unified_and_legacy_render_identical_pages(dataset, tmp_path, limit):
    from pm4bench.io import iter_jsonl
    from pm4bench.mgui.render import render_mgui
    from pm4bench.vision import render_vision

    legacy = tmp_path / 'legacy'
    unified = tmp_path / 'unified'
    render_mgui(dataset / 'metadata/mgui/templates', legacy, languages=('en',))
    report = render_vision(dataset, unified, 'mgui', ('en',), limit=limit,
                           allow_custom_manifest=True)
    assert report['passed'] and report['status'] == 'complete'
    assert report['records'] == limit and report['images'] == 1
    assert len(list(iter_jsonl(unified / 'images.jsonl'))) == limit
    assert (legacy / 'images/example_en.png').read_bytes() == (
        unified / 'images/mgui/en/example_en.png').read_bytes()
    assert (legacy / 'gt/example_en.json').read_bytes() == (
        unified / 'gt/mgui/en/example_en.json').read_bytes()
