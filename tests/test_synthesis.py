import json
from pathlib import Path

import pytest

from pm4bench.qgo.pack import PROMPT, load_split, pack, row_for, split_files
from pm4bench.qgo.synthesize import LANGUAGES, make_html, sample_plan, sort_boxes, validate_plan
from pm4bench.vision.browser import new_output
from pm4bench.vision.mdur import safe_text, validate_style


def test_sampling_is_replayable_and_has_ten_languages():
    words = {language: [f'text-{language}'] for language in LANGUAGES}
    first = sample_plan(words, 0, 42)
    assert first == sample_plan(words, 0, 42)
    assert first != sample_plan(words, 1, 42)
    assert {line['language'] for line in first['lines']} == set(LANGUAGES)
    validate_plan(first)
    assert 'text-en' in make_html(first, '')


def test_ocr_html_is_escaped():
    words = {language: ['<img src=x onerror=alert(1)>'] for language in LANGUAGES}
    document = make_html(sample_plan(words, 0, 42), '')
    assert '&lt;img' in document and '<img src=x' not in document
    assert "script-src 'none'" in document


def test_visual_sort_and_arabic_direction_do_not_mutate_boxes():
    boxes = [{'text': 'B', 'x': 50, 'y': 0, 'height': 20, 'lang': 'en'},
             {'text': 'A', 'x': 10, 'y': 0, 'height': 20, 'lang': 'en'},
             {'text': 'D', 'x': 10, 'y': 40, 'height': 20, 'lang': 'ar'},
             {'text': 'C', 'x': 50, 'y': 40, 'height': 20, 'lang': 'ar'}]
    before = json.dumps(boxes)
    assert [b['text'] for b in sort_boxes(boxes)] == ['A', 'B', 'C', 'D']
    assert json.dumps(boxes) == before


def test_split_is_independent_of_directory_order():
    names = ['b.png', 'a.png', 'c.png']
    assert split_files(names, 1, 42) == split_files(names[::-1], 1, 42)
    with pytest.raises(ValueError):
        split_files(names, 3, 42)


def test_split_manifest_rejects_overlap(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({'train': ['a.png'], 'test': ['a.png']}))
    with pytest.raises(ValueError):
        load_split(path, ['a.png', 'b.png'])


def test_no_input_or_existing_output_overwrite(tmp_path):
    with pytest.raises(ValueError):
        new_output(tmp_path / 'out', tmp_path)
    with pytest.raises(FileExistsError):
        new_output(tmp_path)


def test_legacy_inline_text_is_inert_and_math_is_preserved():
    result = safe_text('x < 20 <image 1><strong onclick="evil()">bold</strong>'
                       '<script>bad()</script><img src="https://evil.test/x">')
    assert result == 'x &lt; 20 <strong>bold</strong>'
    assert safe_text('a\nb', breaks=True) == 'a<br>b'
    assert safe_text(r'$x^2$') == r'$x^2$'


def test_css_injection_rejected():
    with pytest.raises(ValueError):
        validate_style({'size': '32; color:red'})
    words = {language: ['x'] for language in LANGUAGES}
    plan = sample_plan(words, 0, 42)
    plan['id'] = '../escape'
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_mdur_plan_uses_transcript_and_excludes_explanation(tmp_path, monkeypatch):
    from pm4bench.vision import mdur

    assets = ['image_1.png', 'image_2.png']
    for name in assets:
        (tmp_path / name).touch()
    monkeypatch.setattr(mdur, 'routing_metadata', lambda: {
        'en': {'fixture': {'question_images': [1], 'option_images': [[], []],
                           'explanation_images': [2]}}})
    row = {'id': 'fixture', 'source_id': 'fixture', 'language': 'en',
           'question': 'x ', 'options': ['Yes', 'No'],
           'rendered_text': 'Question: x < 20 <image 1>\nOptions:\n(A)  Yes\n(B)  No',
           'traditional_images': assets, 'vision_image': 'reference.png'}
    plan = mdur.mdur_plan(row, tmp_path)
    assert plan['question'] == 'x < 20 <image 1>'
    assert plan['question_images'] == ['image_1.png']
    assert plan['excluded_explanation_images'] == ['image_2.png']


def test_bundled_training_inventory():
    from pm4bench.qgo.synthesize import TEXT_POOL, load_words
    from pm4bench.vision.inputs import sha256

    assert sha256(TEXT_POOL) == '8f8e46bb0415b5fcc67a5c3343a010c3b0d97ab9ea4dafa860464e250d1fdd0c'
    assert all(load_words(TEXT_POOL).values())
    splits = json.loads((TEXT_POOL.parent / 'released_split.json').read_text())
    assert len(splits['train']) == 19500 and len(splits['test']) == 500
    assert len(set(splits['train'] + splits['test'])) == 20000


def test_parquet_pipeline_round_trip(tmp_path):
    pq = pytest.importorskip('pyarrow.parquet')
    from PIL import Image

    source = tmp_path / 'source'
    (source / 'img').mkdir(parents=True)
    (source / 'gt').mkdir()
    for index in range(3):
        image = source / 'img' / f'{index}.png'
        Image.new('RGB', (4, 4), 'white').save(image)
        (source / 'gt' / f'{index}.json').write_text(json.dumps({
            'gt_text': f'text {index}', 'word_boxes': [{'text': f'text {index}'}]}))
    output = tmp_path / 'packed'
    report = pack(source, output, validation_size=1, shard_size=1)
    assert report['train'] == 2 and report['validation'] == 1
    rows = [row for shard in report['shards']
            for row in pq.read_table(output / shard['path']).to_pylist()]
    assert len(rows) == 3
    for row in rows:
        assert row['prompt'] == [{'role': 'user', 'content': PROMPT}]
        assert row['reward_model']['ground_truth'] == row['extra_info']['answer']
        assert not Path(row['extra_info']['image_path']).is_absolute()
        assert row['images'][0]['bytes'].startswith(b'\x89PNG')
    with pytest.raises(ValueError):
        row_for(source / 'img/0.png', {'source': 'gui_v2'}, 'train')
