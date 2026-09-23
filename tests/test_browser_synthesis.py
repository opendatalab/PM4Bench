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
